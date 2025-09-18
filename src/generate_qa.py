import os
from colorama import Fore
import json
from typing import List
from pydantic import BaseModel
from litellm import completion
from prompt_templates import prompt_template

class SimpleChunk:
    """Markdown için basit chunk sınıfı"""
    def __init__(self, text: str, section_name: str = ""):
        self.text = text
        self.section_name = section_name

def extract_text_from_markdown(md_path: str) -> List[SimpleChunk]:
    """Markdown dosyasından metin çıkarma ve chunk'lara bölme"""
    chunks = []
    try:
        with open(md_path, 'r', encoding='utf-8') as file:
            content = file.read()
            print(f"{Fore.CYAN}Markdown dosyası okundu: {len(content)} karakter{Fore.RESET}")
            
            # Metni anlamlı bölümlere ayır
            sections = content.split('\n\n')
            current_section = ""
            current_text = ""
            
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                    
                # Başlık kontrolü
                if section.startswith('#'):
                    # Önceki bölümü kaydet
                    if current_text and len(current_text.strip()) > 100:
                        chunks.append(SimpleChunk(
                            text=current_text.strip(),
                            section_name=current_section
                        ))
                    
                    current_section = section
                    current_text = section + "\n\n"
                else:
                    current_text += section + "\n\n"
                    
                    # Çok uzun bölümleri parçala
                    if len(current_text) > 3000:
                        chunks.append(SimpleChunk(
                            text=current_text.strip(),
                            section_name=current_section
                        ))
                        current_text = ""
            
            # Son bölümü kaydet
            if current_text and len(current_text.strip()) > 100:
                chunks.append(SimpleChunk(
                    text=current_text.strip(),
                    section_name=current_section
                ))
                
    except Exception as e:
        print(f"{Fore.RED}Markdown okuma hatası: {e}{Fore.RESET}")
        return []
    
    print(f"{Fore.GREEN}Toplam {len(chunks)} chunk oluşturuldu{Fore.RESET}")
    return chunks

# Mevcut dataquality.py'den alınan kalite kontrol sınıfları
class Score(BaseModel):
    score: int
    explanation: str

class Rank(BaseModel):
    accuracy: Score
    style: Score

class Record(BaseModel):
    question: str
    answer: str

class Response(BaseModel):
    generated: List[Record]

def llm_call(data: str, num_records: int = 5) -> dict:
    """PDF chunk'ından soru-cevap çiftleri üretir"""
    try:
        stream = completion(
            model="ollama/gemma2:2b",
            messages=[
                {
                    "role": "user",
                    "content": prompt_template(data, num_records),
                }
            ],
            stream=True,
            options={"num_predict": 2000, "temperature": 0.3},
        )
        
        response_data = ""
        for x in stream: 
            delta = x['choices'][0]["delta"]["content"]
            if delta is not None: 
                print(Fore.LIGHTBLUE_EX + delta + Fore.RESET, end="") 
                response_data += delta 
        
        # Boş response kontrolü
        if not response_data.strip():
            print(f"{Fore.YELLOW}Boş response alındı{Fore.RESET}")
            return {"generated": []}
        
        # JSON parsing'i güçlendir
        parsed_data = extract_json_from_response(response_data)
        if parsed_data:
            return {"generated": parsed_data}
        else:
            print(f"{Fore.RED}JSON parse edilemedi{Fore.RESET}")
            return {"generated": []}
            
    except Exception as e:
        print(f"{Fore.RED}LLM çağrısında hata: {e}{Fore.RESET}")
        return {"generated": []}

def extract_json_from_response(response_text: str) -> dict:
    """Response'dan JSON'ı çıkarmaya çalışır - tüm edge case'leri handle eder"""
    
    # 1. Markdown code block'ları temizle
    cleaned = response_text.strip()
    
    # ```json ve ``` temizle
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    
    cleaned = cleaned.strip()
    
    # 2. JSON array veya object bul
    json_start = -1
    
    # [ ile başlayan array ara
    array_start = cleaned.find('[')
    if array_start != -1:
        json_start = array_start
    
    # { ile başlayan object ara (eğer array bulunamadıysa)
    if json_start == -1:
        object_start = cleaned.find('{')
        if object_start != -1:
            json_start = object_start
    
    if json_start == -1:
        print(f"{Fore.YELLOW}JSON start karakteri bulunamadı{Fore.RESET}")
        return None
    
    # 3. JSON'ı parse etmeye çalış
    json_text = cleaned[json_start:]
    
    # 4. JSON'ın nerede bittiğini bul
    try:
        # Önce tam JSON'ı parse etmeye çalış
        return json.loads(json_text)
    except json.JSONDecodeError:
        # Kısmi JSON - en son geçerli JSON'ı bul
        return find_last_valid_json(json_text)

def find_last_valid_json(text: str) -> dict:
    """Text'te en son geçerli JSON'ı bulmaya çalışır"""
    
    # En son } veya ] karakterini bul
    last_brace = text.rfind('}')
    last_bracket = text.rfind(']')
    
    if last_brace == -1 and last_bracket == -1:
        return None
    
    # En son kapanan karakteri kullan
    end_pos = max(last_brace, last_bracket) + 1
    
    # Baştan o pozisyona kadar olan kısmı al
    for start_pos in range(len(text)):
        try:
            json_text = text[start_pos:end_pos]
            result = json.loads(json_text)
            print(f"{Fore.GREEN}JSON bulundu: {start_pos}-{end_pos}{Fore.RESET}")
            return result
        except json.JSONDecodeError:
            continue
    
    return None

def quality_check_qa_pair(qa_pair: dict) -> dict:
    """Tek bir soru-cevap çiftinin kalitesini kontrol eder"""
    record = f"Question: {qa_pair['question']}\nAnswer: {qa_pair['answer']}"
    
    # 3 kez deneme yap
    for attempt in range(3):
        try:
            stream = completion(
                model="ollama/gemma2:2b",
                messages=[
                    {
                        "role": "user",
                        "content": f"""Evaluate this question-answer pair and rate accuracy and style from 1-10.

Question-Answer Pair:
{record}

Rate the pair and return ONLY this JSON format:

```json
{{
  "accuracy": {{
    "score": 8,
    "explanation": "Your explanation here"
  }},
  "style": {{
    "score": 7,
    "explanation": "Your explanation here"
  }}
}}
```

Rating Guidelines:
- Accuracy 1-3: Wrong/incomplete answers
- Accuracy 4-6: Partially correct answers  
- Accuracy 7-10: Correct and complete answers
- Style 1-3: Poor grammar/unclear
- Style 4-6: Average clarity
- Style 7-10: Clear and well-written

IMPORTANT: Return ONLY the JSON, no additional text.""",
                    }
                ],
                stream=True,
                options={"num_predict": 500, "temperature": 0.1},
                timeout=30,  # 30 saniye timeout
            )
        
            data = ""
            for x in stream: 
                delta = x['choices'][0]["delta"]["content"]
                if delta is not None: 
                    data += delta 
            
            # Boş response kontrolü
            if not data.strip():
                print(f"{Fore.YELLOW}Kalite kontrol: Boş response alındı (Deneme {attempt+1}){Fore.RESET}")
                if attempt == 2:  # Son deneme
                    return {
                        "accuracy": {"score": 1, "explanation": "Boş response"},
                        "style": {"score": 1, "explanation": "Boş response"}
                    }
                continue
            
            # JSON parsing'i güçlendir
            parsed_data = extract_json_from_response(data)
            if parsed_data:
                return parsed_data
            else:
                print(f"{Fore.YELLOW}Kalite kontrol JSON parse edilemedi (Deneme {attempt+1}){Fore.RESET}")
                if attempt == 2:  # Son deneme
                    return {
                        "accuracy": {"score": 1, "explanation": "JSON parse hatası"},
                        "style": {"score": 1, "explanation": "JSON parse hatası"}
                    }
                continue
                
        except Exception as e:
            print(f"{Fore.YELLOW}Kalite kontrolünde hata (Deneme {attempt+1}): {e}{Fore.RESET}")
            if attempt == 2:  # Son deneme
                return {
                    "accuracy": {"score": 1, "explanation": f"Error: {str(e)}"},
                    "style": {"score": 1, "explanation": f"Error: {str(e)}"}
                }
            import time
            time.sleep(2)  # 2 saniye bekle
            continue
    
    # Hiçbir deneme başarılı olmadı
    return {
        "accuracy": {"score": 1, "explanation": "Tüm denemeler başarısız"},
        "style": {"score": 1, "explanation": "Tüm denemeler başarısız"}
    }

def process_markdown_to_qa(md_path: str, output_file: str = "markdown_qa_dataset.json", quality_output_file: str = "markdown_qa_quality.json", min_quality_score: int = 6):
    """
    Markdown dosyasını işleyip kaliteli soru-cevap çiftleri üretir
    """
    print(f"{Fore.GREEN}Markdown işleniyor: {md_path}{Fore.RESET}")
    
    # Markdown'dan chunk'ları çıkar
    chunks = extract_text_from_markdown(md_path)
    
    if not chunks:
        print(f"{Fore.RED}Markdown dosyasından chunk çıkarılamadı!{Fore.RESET}")
        return
    
    print(f"{Fore.YELLOW}Toplam {len(chunks)} chunk bulundu{Fore.RESET}")
    
    # Dosyaları başlat
    high_quality_pairs = []
    quality_results = []
    
    # Her chunk'tan 3 QA üret, hemen kalite kontrol et ve dosyaya yaz
    print(f"{Fore.CYAN}🚀 Chunk bazında QA üretimi ve kalite kontrol başlıyor...{Fore.RESET}")
    
    for i, chunk in enumerate(chunks):
        print(f"\n{Fore.CYAN}Chunk {i+1}/{len(chunks)} işleniyor...{Fore.RESET}")
        if chunk.section_name:
            print(f"{Fore.LIGHTMAGENTA_EX}Bölüm: {chunk.section_name[:50]}...{Fore.RESET}")
        print(f"{Fore.YELLOW}Metin Uzunluğu: {len(chunk.text)} karakter{Fore.RESET}")
        
        # Her chunk'tan 3 soru-cevap üret
        generated_data = llm_call(chunk.text, num_records=3)
        
        if "generated" in generated_data:
            chunk_qa_pairs = generated_data["generated"]
            print(f"{Fore.GREEN}✓ {len(chunk_qa_pairs)} soru-cevap çifti üretildi{Fore.RESET}")
            
            # Hemen kalite kontrol et
            print(f"{Fore.YELLOW}🔍 Kalite kontrol başlıyor...{Fore.RESET}")
            chunk_high_quality = []
            chunk_quality_results = []
            
            for qa_pair in chunk_qa_pairs:
                quality_assessment = quality_check_qa_pair(qa_pair)
                
                # Güvenli score çıkarma
                try:
                    if isinstance(quality_assessment, dict):
                        if 'accuracy' in quality_assessment and 'style' in quality_assessment:
                            accuracy_score = quality_assessment['accuracy']['score'] if isinstance(quality_assessment['accuracy'], dict) else quality_assessment['accuracy']
                            style_score = quality_assessment['style']['score'] if isinstance(quality_assessment['style'], dict) else quality_assessment['style']
                        else:
                            accuracy_score = 1
                            style_score = 1
                    else:
                        accuracy_score = 1
                        style_score = 1
                except Exception as e:
                    print(f"{Fore.YELLOW}Kalite score hatası: {e}{Fore.RESET}")
                    accuracy_score = 1
                    style_score = 1
                
                chunk_quality_results.append({**qa_pair, 'quality': quality_assessment})
                
                if accuracy_score >= min_quality_score and style_score >= min_quality_score:
                    chunk_high_quality.append(qa_pair)
                    print(f"{Fore.GREEN}✓ Yüksek kalite (A:{accuracy_score}, S:{style_score}){Fore.RESET}")
                else:
                    print(f"{Fore.RED}✗ Düşük kalite (A:{accuracy_score}, S:{style_score}){Fore.RESET}")
            
            # Hemen dosyaya ekle
            high_quality_pairs.extend(chunk_high_quality)
            quality_results.extend(chunk_quality_results)
            
            # Dosyaları güncelle
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(high_quality_pairs, f, ensure_ascii=False, indent=4)
            
            with open(quality_output_file, 'w', encoding='utf-8') as f:
                json.dump(quality_results, f, ensure_ascii=False, indent=4)
            
            print(f"{Fore.GREEN}💾 Dosyalar güncellendi! Toplam {len(high_quality_pairs)} yüksek kaliteli QA{Fore.RESET}")
    
    print(f"\n{Fore.GREEN}🎉 TÜM İŞLEMLER TAMAMLANDI!{Fore.RESET}")
    
    # İstatistikler
    print(f"\n{Fore.CYAN}📊 İSTATİSTİKLER:{Fore.RESET}")
    print(f"Toplam chunk: {len(chunks)}")
    print(f"Toplam üretilen QA: {len(quality_results)}")
    print(f"Yüksek kaliteli QA: {len(high_quality_pairs)}")
    
    if len(quality_results) > 0:
        print(f"Başarı oranı: {(len(high_quality_pairs)/len(quality_results)*100):.1f}%")
    else:
        print(f"Başarı oranı: 0% (Hiç QA üretilmedi)")
    
    print(f"\n{Fore.GREEN}✅ Yüksek kaliteli soru-cevap çiftleri '{output_file}' dosyasına kaydedildi.{Fore.RESET}")
    print(f"{Fore.GREEN}✅ Tüm kalite değerlendirme sonuçları '{quality_output_file}' dosyasına kaydedildi.{Fore.RESET}")

if __name__ == "__main__":
    # Markdown'dan soru-cevap üretme ve kalite kontrolü
    # Örnek kullanım: Betiği src klasöründen çalıştırırken, data klasörüne ../ ile erişilir.
    # Ana dizinden çalıştırmak için (örn: python src/generate_qa.py) yolların başına ../ koyun.
    process_markdown_to_qa(
        md_path="..\data\raw\using-manufacturing.md",  # Ham veri klasöründeki dosyayı okur
        output_file="..\data\processed\generated_qa.json", # İşlenmiş veriyi kaydeder
        quality_output_file="..\data\processed\generated_qa_report.json", # Raporu kaydeder
        min_quality_score=6
    )
