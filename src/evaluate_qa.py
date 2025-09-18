import json
from pydantic import BaseModel
from litellm import completion
from colorama import Fore 
import time

class Score(BaseModel):
    score: int
    explanation: str

class Rank(BaseModel):
    accuracy: Score
    style: Score

def extract_json_from_response(response_text: str) -> dict:
    """Response'dan JSON'ı çıkarmaya çalışır"""
    # Markdown code block'ları temizle
    cleaned = response_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    
    # JSON object bul
    json_start = cleaned.find('{')
    if json_start != -1:
        json_text = cleaned[json_start:]
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            return None
    return None

def llm_call(record: dict) -> dict:
    """Final kalite kontrol için gelişmiş LLM çağrısı"""
    question = record.get('question', '')
    answer = record.get('answer', '')
    record_str = f"Question: {question}\nAnswer: {answer}"
    
    # 3 kez deneme yap
    for attempt in range(3):
        try:
            stream = completion(
                model="ollama/gemma2:2b",
                messages=[
                    {
                        "role": "user",
                        "content": f"""FINAL QUALITY CHECK: Rate this instruction tuning record from 1-10 for accuracy and style.

Record:
{record_str}

Evaluate based on these strict criteria:

ACCURACY (1-10):
- 9-10: Perfect, comprehensive, technically accurate
- 7-8: Very good, mostly accurate with minor gaps
- 5-6: Adequate, generally correct but incomplete
- 3-4: Poor, significant inaccuracies
- 1-2: Wrong, misleading, or blank

STYLE (1-10):
- 9-10: Excellent clarity, professional, well-structured
- 7-8: Good clarity, minor style issues
- 5-6: Adequate readability, some confusion
- 3-4: Poor clarity, hard to follow
- 1-2: Very poor, unprofessional, or blank

Return ONLY this JSON format:

```json
{{
  "accuracy": {{
    "score": 8,
    "explanation": "Detailed explanation of accuracy rating"
  }},
  "style": {{
    "score": 7,
    "explanation": "Detailed explanation of style rating"
  }}
}}
```

IMPORTANT: Return ONLY the JSON, no additional text.""",
                    }
                ],
                stream=True,
                options={"num_predict": 800, "temperature": 0.1},
                timeout=30,
            )
            
            data = ""
            for x in stream: 
                delta = x['choices'][0]["delta"]["content"]
                if delta is not None: 
                    data += delta 
            
            # JSON parse et
            parsed_data = extract_json_from_response(data)
            if parsed_data:
                return parsed_data
            else:
                print(f"{Fore.YELLOW}JSON parse hatası (Deneme {attempt+1}){Fore.RESET}")
                if attempt == 2:
                    return {
                        "accuracy": {"score": 1, "explanation": "JSON parse hatası"},
                        "style": {"score": 1, "explanation": "JSON parse hatası"}
                    }
                continue
                
        except Exception as e:
            print(f"{Fore.YELLOW}LLM hatası (Deneme {attempt+1}): {e}{Fore.RESET}")
            if attempt == 2:
                return {
                    "accuracy": {"score": 1, "explanation": f"Error: {str(e)}"},
                    "style": {"score": 1, "explanation": f"Error: {str(e)}"}
                }
            time.sleep(2)
            continue
    
    return {
        "accuracy": {"score": 1, "explanation": "Tüm denemeler başarısız"},
        "style": {"score": 1, "explanation": "Tüm denemeler başarısız"}
    }


if __name__ == "__main__": 
    print(f"{Fore.GREEN}🔍 FINAL KALİTE KONTROL BAŞLIYOR...{Fore.RESET}")
    
    # İlk aşamadan gelen yüksek kaliteli QA'ları oku
    input_file = '..\data\processed\generated_qa.json'  # Birinci aşamadan gelen dosya
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f: 
            data = json.load(f)
        print(f"{Fore.CYAN}📂 {len(data)} QA çifti yüklendi{Fore.RESET}")
    except FileNotFoundError:
        print(f"{Fore.RED}❌ {input_file} dosyası bulunamadı!{Fore.RESET}")
        exit(1)
    
    final_quality_pairs = []
    all_quality_results = []
    
    print(f"{Fore.YELLOW}🎯 Final kalite kontrolü için minimum score: 8{Fore.RESET}")
    
    for i, pair in enumerate(data): 
        print(f"\n{Fore.CYAN}QA {i+1}/{len(data)} kontrol ediliyor...{Fore.RESET}")
        print(f"{Fore.LIGHTBLUE_EX}Q: {pair['question'][:80]}...{Fore.RESET}")
        
        result = llm_call(pair) 
        accuracy_score = result['accuracy']['score']
        style_score = result['style']['score']
        
        all_quality_results.append({**pair, 'final_quality': result})
        
        # Çok yüksek kalite threshold: 8+ 
        if accuracy_score >= 8 and style_score >= 8:
            final_quality_pairs.append(pair)
            print(f"{Fore.GREEN}✅ SÜPER KALİTE (A:{accuracy_score}, S:{style_score}){Fore.RESET}")
        elif accuracy_score >= 7 and style_score >= 7:
            print(f"{Fore.YELLOW}✓ İyi kalite (A:{accuracy_score}, S:{style_score}){Fore.RESET}")
        else:
            print(f"{Fore.RED}✗ Düşük kalite (A:{accuracy_score}, S:{style_score}){Fore.RESET}")
    
    # Sonuçları kaydet
    print(f"\n{Fore.GREEN}💾 Sonuçlar kaydediliyor...{Fore.RESET}")
    
    # Final yüksek kaliteli QA'lar
    with open('..\data\processed\final_premium_qa.json','w', encoding='utf-8') as f: 
        json.dump(final_quality_pairs, f, ensure_ascii=False, indent=4)
    
    # Tüm final kalite sonuçları
    with open('..\data\processed\final_quality_results.json','w', encoding='utf-8') as f: 
        json.dump(all_quality_results, f, ensure_ascii=False, indent=4)
    
    # İstatistikler
    print(f"\n{Fore.CYAN}📊 FINAL İSTATİSTİKLER:{Fore.RESET}")
    print(f"Başlangıç QA sayısı: {len(data)}")
    print(f"Final premium QA sayısı: {len(final_quality_pairs)}")
    print(f"Premium başarı oranı: {(len(final_quality_pairs)/len(data)*100):.1f}%")
    
    print(f"\n{Fore.GREEN}✅ Final premium QA'lar: 'data/processed/final_premium_qa.json'{Fore.RESET}")
    print(f"{Fore.GREEN}✅ Final kalite sonuçları: 'data/processed/final_quality_results.json'{Fore.RESET}")