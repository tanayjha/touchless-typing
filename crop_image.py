from PIL import Image
im = Image.open('test0.jpeg')

left = 450
top = 150
right = 650
bottom = 450

im.crop((left, top, right, bottom)).save('cropped_test.jpeg')