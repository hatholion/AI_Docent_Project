# 사용 모델

## embedding model

|모델명|특징|크기|
|-|-|-|
|nlpai-lab/KURE-v1|한국어 retrieval 특화|0.6B|
|BAAI/bge-m3|다국어 가장 많이 쓰는 모델|0.6B|
|dragonkue/BGE-m3-ko|bge-m3를 한국어로 특화|0.6B|
|intfloat/multilingual-e5-large|다국어 가장 많이 쓰는 모델|0.6B|
|jhgan/ko-sroberta-multitask|한국어 SBERT 계열 고전 baseline|0.1B|



## LLM model
| 모델명 | 파라미터 | 특징 | 언어 | HF 원본 dtype | GPU weight 용량 | Hugging Face | Ollama | Ollama 용량 |
|---|---:|---|---|---|---:|---|---|---:|
| **EXAONE 3.5 2.4B** | 2.4B급 (실제 약 2.67B) | LG AI Research. 경량 instruction 모델, 32K context. 한국어·영어를 주 언어로 학습 | **한국어, 영어** | **FP32** | **약 9.6GB** | `LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct` | `exaone3.5:2.4b-instruct-q4_K_M` | **1.6GB (Q4_K_M)** |
| **EXAONE 3.5 7.8B** | 7.8B급 (약 7.82B) | EXAONE 3.5 중 중형 모델. 한국어·영어 bilingual, 32K context | **한국어, 영어** | **FP32** | **약 31.3GB** | `LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct` :chatgpt-content-reference{index="2"} | `exaone3.5:7.8b-instruct-q4_K_M` | **4.8GB (Q4_K_M)** |
| **Bllossom 3B** | 약 3.21B | Llama 3.2 3B 기반. 원본 Llama 3.2에 부족했던 한국어를 **150GB 정제 한국어 데이터로 추가 사전학습**, instruction tuning 적용 | **한국어, 영어** | **BF16** | **약 6.44GB** | `Bllossom/llama-3.2-Korean-Bllossom-3B` :chatgpt-content-reference{index="4"} | `timHan/llama3.2korean3B4QKM` | **2.0GB (Q4_K_M)** :chatgpt-content-reference{index="5"} |
| **Bllossom 8B** | 약 8.03B | Llama 3 8B 기반 한국어 강화 모델. 한국어/영어 생성용 | **한국어, 영어** | **BF16** | **약 16.1GB** | `MLP-KTLim/llama-3-Korean-Bllossom-8B` | `timHan/llama3korean8B4QKM` | **4.9GB (Q4_K_M)** |
| **DNA 1.0 8B** | 약 8.03B | Llama 계열. 한국어 CPT + Knowledge Distillation + SFT + DPO 적용. **한국어 이해·생성에 명시적으로 최적화** | **한국어, 영어** | **BF16** | **약 16.1GB** | `dnotitia/Llama-DNA-1.0-8B-Instruct` | `dnotitia/dna:8b-instruct-q4_K` | **4.9GB (Q4_K_M)** |
| **Ko-Gemma 2 9B** | 약 9.24B | Google Gemma 2 9B 기반. 한국어 고품질 데이터로 SFT 후 DPO 적용한 **한국어 대화 특화 모델** | **한국어 중심** | **BF16** | **약 18.5GB** | `rtzr/ko-gemma-2-9b-it`| `architectyou/ko-gemma2-9B-Q4_K_M` | **5.8GB (Q4_K_M)** |
| **Gemma 4 E2B** | **2.3B effective / 5.1B total** | Google 최신 Gemma 4 경량 모델. reasoning, function calling, 멀티모달, 128K context. 한국어 특화는 아니고 다국어 범용 | **다국어** | **BF16** | **약 10.2GB** | `google/gemma-4-E2B-it`| `gemma4:e2b` | **7.2GB (Q4_K_M)** :chatgpt-content-reference{index="13"} |
