import cv2

# Carregar o classificador pré-treinado para detecção de rostos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Carregar a imagem
image_path = 'sua_imagem.jpg'  # Substitua pelo caminho da sua imagem
image = cv2.imread(image_path)

if image is None:
    print("Imagem não encontrada. Verifique o caminho.")
    exit()

# Converter a imagem para escala de cinza
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detectar rostos na imagem
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Desenhar retângulos ao redor dos rostos detectados
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Mostrar a imagem com os rostos detectados
cv2.imshow('Detecção de Rostos', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
