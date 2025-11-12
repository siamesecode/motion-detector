from PIL import Image, ImageDraw, ImageFont
import os


width, height = 400, 400
img = Image.new('RGB', (width, height))
draw = ImageDraw.Draw(img)


for i in range(height):
    color_r = int(255 * i / height)
    color_b = 255 - color_r
    draw.line([(0, i), (width, i)], fill=(color_r, 100, color_b))

draw.rectangle([100, 50, 300, 80], fill=(255, 255, 0))
draw.ellipse([150, 270, 250, 370], fill=(0, 255, 0))


try:
    font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
except:
    font_large = ImageFont.load_default()
    font_medium = ImageFont.load_default()

draw.text((80, 150), 'SUCESSO!', fill=(255, 255, 255), font=font_large)
draw.text((50, 230), 'Maos Levantadas', fill=(255, 255, 255), font=font_medium)


os.makedirs('images', exist_ok=True)
img.save('images/sample_image.jpg')
print("Imagem de exemplo criada com sucesso!")
