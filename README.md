1. Crie um ambiente virtual
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Instale as dependências
```bash
pip install -r requirements.txt
```

## Como usar

1. Coloque suas imagens na pasta `images/`
2. Execute o programa:
```bash
python3 hand_gesture_image.py
```

3. Faça os gestos em frente à webcam
4. Pressione 'q' para sair


## Tecnologias

- OpenCV - Processamento de vídeo
- MediaPipe - Detecção de mãos
- NumPy - Manipulação de dados

