import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Verificação de bibliotecas multimediáticas avançadas
try:
    import easyocr
    import whisper
    MODULOS_REAIS_DISPONIVEIS = True
except ImportError:
    MODULOS_REAIS_DISPONIVEIS = False
    print("\n[AVISO] 'easyocr' ou 'openai-whisper' não detetados. O sistema utilizará o modo de simulação por prefixo.")

# 1. CARREGAMENTO E BASE DE CONHECIMENTO (RAG)
print("\n" + "="*60)
print("ASSISTENTE INTELEGENE MULTIMODAL DE CHAMADOS")
print("="*60)

print("Carregando modelo de Embeddings...")
modelo_embed = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Inicialização de leitores de Imagem e Áudio
reader_ocr = None
modelo_whisper = None

if MODULOS_REAIS_DISPONIVEIS:
    print("Inicializando leitor de Imagens (EasyOCR)...")
    reader_ocr = easyocr.Reader(['pt', 'en'], gpu=False)
    print("Inicializando transcritor de Áudio (Whisper)...")
    modelo_whisper = whisper.load_model("tiny")

# Base de Conhecimento Oficial (RAG)
base_documentos = [
    {
        "id": "DOC01", 
        "pergunta": "Como solicitar segunda chamada de prova, atestado médico ou justificativa de ausência em prova?", 
        "resposta": "Acesse o portal do aluno na aba Requerimentos e anexe o atestado em até 48h."
    },
    {
        "id": "DOC02", 
        "pergunta": "O sistema, portal acadêmico, Wi-Fi, login ou notas estão fora do ar ou com erro de acesso e conexão", 
        "resposta": "Reinicie a conexão ou abra um chamado de TI no portal para verificação de conta."
    },
    {
        "id": "DOC03", 
        "pergunta": "Onde emitir ou pagar boleto de mensalidade em atraso, chave pix ou consulta financeira?", 
        "resposta": "Os boletos atualizados podem ser gerados diretamente no Portal Financeiro."
    },
    {
        "id": "DOC04", 
        "pergunta": "Projetor não liga, sala de aula, lâmpada queimada, luz de erro, cabo solto ou equipamento físico quebrado", 
        "resposta": "A equipe de suporte presencial de Infraestrutura deve ser acionada no ramal 2004 para substituição imediata do equipamento."
    }
]

textos_base = [f"{d['pergunta']} {d['resposta']}" for d in base_documentos]
vetores_base = modelo_embed.encode(textos_base, normalize_embeddings=True)

# 2. TREINAMENTO DO CLASSIFICADOR (Random Forest)
X_treino = [
    "não consigo acessar o portal acadêmico erro de conexão login wifi notas",
    "sistema fora do ar erro de senha portal ver minhas notas",
    "cabo de rede partido no laboratório de informática",
    "projetor da sala 204 não liga e a aula começa em 10 minutos",
    "projetor queimado e fumegando luz vermelha de erro",
    "lâmpada do projetor queimou luz de erro ligada",
    "ar condicionado quebrado vazando água na sala",
    "equipamento da sala parou de funcionar durante a aula",
    "preciso do meu boleto de mensalidade atrasado pagar conta",
    "valor da faculdade pagamento chave pix financeiro emitir boleto",
    "segunda chamada de prova por atestado médico faltei na prova",
    "faltei na prova como justificativa ausência perdi a prova"
]

y_categoria = [
    "TI/Sistemas", "TI/Sistemas", "TI/Sistemas",
    "Infraestrutura", "Infraestrutura", "Infraestrutura", "Infraestrutura", "Infraestrutura",
    "Financeiro", "Financeiro",
    "Acadêmico", "Acadêmico"
]

y_prioridade = [
    "Alta", "Média", "Média",
    "Alta", "Alta", "Alta", "Média", "Alta",
    "Baixa", "Baixa",
    "Média", "Baixa"
]

vectorizer = TfidfVectorizer()
X_vetorizado = vectorizer.fit_transform(X_treino)

clf_categoria = RandomForestClassifier(random_state=42).fit(X_vetorizado, y_categoria)
clf_prioridade = RandomForestClassifier(random_state=42).fit(X_vetorizado, y_prioridade)

# 3. ROTEADOR DE ENTRADA MULTIMODAL
def roteador_entrada(entrada):
    caminho_str = entrada.strip("'\"")
    extensoes_audio = ['.mp3', '.wav', '.m4a', '.ogg', '.flac']
    extensoes_imagem = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    
    _, ext = os.path.splitext(caminho_str.lower())

    # Processamento de Ficheiro de Áudio Real
    if os.path.isfile(caminho_str) and ext in extensoes_audio:
        modalidade = "Áudio / Voz (Transcrição Whisper STT)"
        if modelo_whisper:
            res = modelo_whisper.transcribe(caminho_str, language="pt")
            texto_limpo = res["text"].strip()
        else:
            texto_limpo = "Instale openai-whisper para processar ficheiros de áudio reais."

    # Processamento de Ficheiro de Imagem Real
    elif os.path.isfile(caminho_str) and ext in extensoes_imagem:
        modalidade = "Imagem Visual (Análise OCR/EasyOCR)"
        if reader_ocr:
            resultados = reader_ocr.readtext(caminho_str, detail=0)
            texto_limpo = " ".join(resultados).strip()
            if not texto_limpo:
                texto_limpo = "Imagem sem texto legível identificável."
        else:
            texto_limpo = "Instale easyocr para processar ficheiros de imagem reais."

    # Simulação por Prefixes
    elif caminho_str.lower().startswith("audio:"):
        modalidade = "Áudio / Voz (Transcrição STT)"
        texto_limpo = caminho_str.replace("audio:", "", 1).replace("Audio:", "", 1).strip()

    elif caminho_str.lower().startswith("foto:") or caminho_str.lower().startswith("imagem:"):
        modalidade = "Imagem Visual (Análise VLM/OCR)"
        texto_limpo = caminho_str.replace("foto:", "", 1).replace("imagem:", "", 1).strip()

    # Texto Direto
    else:
        modalidade = "Texto Direto"
        texto_limpo = caminho_str.strip()

    return modalidade, texto_limpo

# 4. PIPELINE DE PROCESSAMENTO DO CHAMADO
def processar_chamado(entrada_bruta):
    modalidade, texto_processado = roteador_entrada(entrada_bruta)
    
    # Previsão ML
    vetor_input = vectorizer.transform([texto_processado])
    cat_pred = clf_categoria.predict(vetor_input)[0]
    prio_pred = clf_prioridade.predict(vetor_input)[0]
    
    # Extração de palavras-chave para Justificativa
    stopwords = {
        "para", "como", "esta", "com", "minhas", "minha", "pelo", "pela", 
        "imagem", "audio", "foto", "preciso", "quero", "sobre", "onde",
        "mais", "muito", "estou", "está", "isso", "esse", "essa"
    }
    palavras_chave = [w.strip(".,:;!?") for w in texto_processado.split() if len(w) > 3 and w.lower() not in stopwords]
    if not palavras_chave:
        palavras_chave = texto_processado.split()

    justificativa = f"Escolhi a categoria '{cat_pred}' com prioridade '{prio_pred}' baseando-me nos termos '{', '.join(palavras_chave[:3])}'."

    # RAG - Busca Semântica
    vetor_pergunta = modelo_embed.encode([texto_processado], normalize_embeddings=True)
    similaridades = np.dot(vetores_base, vetor_pergunta.T).squeeze()
    
    idx_melhor = int(np.argmax(similaridades))
    score_melhor = float(similaridades[idx_melhor])
    
    # Limiar de Confiabilidade
    if score_melhor >= 0.35:
        doc_relacionado = base_documentos[idx_melhor]["id"]
        resposta = base_documentos[idx_melhor]["resposta"]
    else:
        doc_relacionado = "Nenhum documento exato encontrado"
        resposta = "Não foi encontrada uma resposta exata na base oficial. Chamado encaminhado para atendimento humano."
        
    return {
        "Entrada Bruta": entrada_bruta,
        "Modalidade Identificada": modalidade,
        "Texto Processado": texto_processado,
        "Saídas Esperadas": {
            "1. Categoria": cat_pred,
            "2. Prioridade": prio_pred,
            "3. Documentos Relacionados": [doc_relacionado],
            "4. Resposta Fundamentada": resposta,
            "Score de Confiabilidade RAG": f"{score_melhor:.3f}",
            "Justificativa Automática": justificativa
        }
    }

# 5. EXECUÇÃO
print("\nAssistente iniciado com sucesso. Envie seu chamado abaixo:")
while True:
    print("\n" + "-"*60)
    entrada = input("Envie o chamado (Texto, Caminho da Foto/Áudio ou Prefixo): ").strip()
    
    if entrada.lower() in ["sair", "exit", "0"]:
        print("\nEncerrando o assistente. Até logo!")
        break
        
    if not entrada:
        continue
        
    resultado = processar_chamado(entrada)
    print("\nRESULTADO DO ATENDIMENTO:")
    print(json.dumps(resultado, indent=2, ensure_ascii=False))
