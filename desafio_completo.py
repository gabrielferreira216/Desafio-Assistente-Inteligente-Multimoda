import os
import json
import numpy as np
import tkinter as tk
from tkinter import filedialog
import cv2
import sounddevice as sd
from scipy.io.wavfile import write
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Importação dos leitores de IA
import easyocr
import whisper

# 1. CARREGAMENTO DOS MODELOS
print("\n" + "="*60)
print("INICIALIZANDO ASSISTENTE MULTIMODAL INTERATIVO")
print("="*60)

print("1/3 - Carregando RAG (Embeddings)...")
modelo_embed = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

print("2/3 - Carregando OCR (Leitor de Imagem EasyOCR)...")
reader_ocr = easyocr.Reader(['pt', 'en'], gpu=False)

print("3/3 - Carregando STT (Transcritor de Voz Whisper)...")
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

# 3. CAPTURADORES REAIS (CAMERA, MICROFONE, EXPLORADOR)

def abrir_janela_arquivo():
    """Abre o Explorador de Arquivos do Windows para escolher um arquivo"""
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    caminho = filedialog.askopenfilename(
        title="Selecione a Imagem, Áudio ou Texto do chamado",
        filetypes=[
            ("Todos os Arquivos Suportados", "*.png *.jpg *.jpeg *.bmp *.mp3 *.wav *.m4a *.txt"),
            ("Imagens", "*.png *.jpg *.jpeg *.bmp"),
            ("Áudios", "*.mp3 *.wav *.m4a"),
            ("Texto", "*.txt")
        ]
    )
    return caminho

def gravar_audio_microfone(segundos=5, taxa_amostragem=44100):
    """Grava o som do microfone por N segundos"""
    print(f"\n🎙️ GRAVANDO MICROFONE POR {segundos} SEGUNDOS... FALE AGORA!")
    gravacao = sd.rec(int(segundos * taxa_amostragem), samplerate=taxa_amostragem, channels=1, dtype='int16')
    sd.wait()
    print("✅ Gravação concluída!")
    
    caminho_audio = "audio_gravado.wav"
    write(caminho_audio, taxa_amostragem, gravacao)
    return caminho_audio

def capturar_foto_webcam():
    """Abre a webcam. Pressione ESPAÇO para tirar a foto ou ESC para cancelar."""
    print("\n📷 ABRINDO WEBCAM... Pressione [ESPAÇO] para tirar a foto ou [ESC] para sair.")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Erro ao acessar a webcam.")
        return None

    caminho_foto = None
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Erro ao ler a imagem da câmera.")
            break

        cv2.imshow("Webcam - Pressione ESPACO para capturar", frame)
        key = cv2.waitKey(1)
        
        if key % 256 == 27:  # ESC
            print("Captura cancelada.")
            break
        elif key % 256 == 32:  # ESPAÇO
            caminho_foto = "foto_webcam.jpg"
            cv2.imwrite(caminho_foto, frame)
            print("✅ Foto tirada com sucesso!")
            break

    cap.release()
    cv2.destroyAllWindows()
    return caminho_foto

# 4. ROTEADOR MULTIMODAL REAL
def processar_entrada(caminho_ou_texto, modo):
    if modo == "texto":
        return "Texto Direto", caminho_ou_texto

    extensoes_audio = ['.mp3', '.wav', '.m4a', '.ogg', '.flac']
    extensoes_imagem = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    _, ext = os.path.splitext(caminho_ou_texto.lower())

    # Processamento de Áudio
    if ext in extensoes_audio or modo == "audio":
        print(f"-> Transcrevendo áudio via Whisper...")
        res = modelo_whisper.transcribe(caminho_ou_texto, language="pt")
        texto_extraido = res["text"].strip()
        modalidade = "Áudio / Voz (Microfone/Arquivo STT)"

    # Processamento de Imagem
    elif ext in extensoes_imagem or modo == "camera":
        print(f"-> Lendo texto da imagem via EasyOCR...")
        resultados = reader_ocr.readtext(caminho_ou_texto, detail=0)
        texto_extraido = " ".join(resultados).strip()
        if not texto_extraido:
            texto_extraido = "Imagem capturada sem texto legível identificável"
        modalidade = "Imagem Visual (Webcam/Arquivo OCR)"

    else:
        # Ficheiro de Texto (.txt)
        with open(caminho_ou_texto, 'r', encoding='utf-8') as f:
            texto_extraido = f.read().strip()
        modalidade = "Arquivo de Texto (.txt)"

    return modalidade, texto_extraido

# 5. PIPELINE COMPLETO
def processar_chamado(caminho_ou_texto, modo):
    modalidade, texto_processado = processar_entrada(caminho_ou_texto, modo)
    
    print(f"\nTexto Extraído: '{texto_processado}'")
    
    # Classificação ML
    vetor_input = vectorizer.transform([texto_processado])
    cat_pred = clf_categoria.predict(vetor_input)[0]
    prio_pred = clf_prioridade.predict(vetor_input)[0]
    
    # Justificativa
    stopwords = {"para", "como", "esta", "com", "minhas", "minha", "pelo", "pela", "preciso", "quero", "sobre"}
    palavras_chave = [w.strip(".,:;!?") for w in texto_processado.split() if len(w) > 3 and w.lower() not in stopwords]
    if not palavras_chave:
        palavras_chave = texto_processado.split()

    justificativa = f"Escolhi a categoria '{cat_pred}' com prioridade '{prio_pred}' baseando-me nos termos '{', '.join(palavras_chave[:3])}'."

    # RAG
    vetor_pergunta = modelo_embed.encode([texto_processado], normalize_embeddings=True)
    similaridades = np.dot(vetores_base, vetor_pergunta.T).squeeze()
    
    idx_melhor = int(np.argmax(similaridades))
    score_melhor = float(similaridades[idx_melhor])
    
    if score_melhor >= 0.35:
        doc_relacionado = base_documentos[idx_melhor]["id"]
        resposta = base_documentos[idx_melhor]["resposta"]
    else:
        doc_relacionado = "Nenhum documento exato encontrado"
        resposta = "Não foi encontrada uma resposta exata na base oficial. Chamado encaminhado para atendimento humano."
        
    return {
        "Entrada Bruta": caminho_ou_texto,
        "Modalidade Identificada": modalidade,
        "Texto Processado Extraído": texto_processado,
        "Saídas Esperadas": {
            "1. Categoria": cat_pred,
            "2. Prioridade": prio_pred,
            "3. Documentos Relacionados": [doc_relacionado],
            "4. Resposta Fundamentada": resposta,
            "Score de Confiabilidade RAG": f"{score_melhor:.3f}",
            "Justificativa Automática": justificativa
        }
    }

# 6. MENU INTERATIVO PRINCIPAL
while True:
    print("\n" + "="*60)
    print("MENU DE ATENDIMENTO AO CLIENTE - SELECIONE O MODO")
    print("="*60)
    print("[ 1 ] Digitar Texto do Chamado")
    print("[ 2 ] Escolher Arquivo no Computador (Janela do Windows)")
    print("[ 3 ] Gravar Áudio Agora pelo Microfone")
    print("[ 4 ] Tirar Foto Agora pela Câmera (Webcam)")
    print("[ 0 ] Sair")
    
    opcao = input("\nEscolha uma opção (0-4): ").strip()
    
    caminho_ou_texto = None
    modo = None

    if opcao == "1":
        caminho_ou_texto = input("\nDigite a sua dúvida ou problema: ").strip()
        modo = "texto"
    
    elif opcao == "2":
        print("\nAbrindo janela de arquivos...")
        caminho_ou_texto = abrir_janela_arquivo()
        modo = "arquivo"
        if not caminho_ou_texto:
            print("Nenhum arquivo selecionado.")
            continue
            
    elif opcao == "3":
        segundos = int(input("Quantos segundos deseja gravar? (Ex: 5): ") or "5")
        caminho_ou_texto = gravar_audio_microfone(segundos=segundos)
        modo = "audio"
        
    elif opcao == "4":
        caminho_ou_texto = capturar_foto_webcam()
        modo = "camera"
        if not caminho_ou_texto:
            continue
            
    elif opcao in ["0", "sair", "exit"]:
        print("\nEncerrando o sistema. Até logo!")
        break
    else:
        print("Opção inválida, tente novamente.")
        continue

    if caminho_ou_texto:
        resultado = processar_chamado(caminho_ou_texto, modo)
        print("\nRESULTADO DO ATENDIMENTO:")
        print(json.dumps(resultado, indent=2, ensure_ascii=False))
