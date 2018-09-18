from PIL import Image, ImageDraw, ImageFont
import random, string
import numpy as np

class Generator():
	def __init__(self, length, size, margin, noise):
		self.length = length
		self.fnt = ImageFont.truetype('arial.ttf', 9)
		self.size = size
		self.margin = np.array(margin)
		self.noise = noise

	def generate(self):
		img = Image.new('L', self.size)

		length = np.random.randint(1, self.length)
		chars = [random.choice(string.ascii_uppercase) for i in range(length)]
		txt = "".join(chars)

		d = ImageDraw.Draw(img)
		txtsize = d.textsize(txt, font=self.fnt)
		pos = np.random.rand(2) * (self.size - 2 * self.margin - txtsize) + self.margin
		d.text(pos, txt, font=self.fnt, fill=128)
		 
		noise = self.noise * np.random.randn(*self.size) * 255
		img = img + noise
		# img = Image.fromarray(img)

		chars.extend(['['] * (self.length - length)) # ending character
		return chars, img, length

if __name__ == "__main__":
	gen = Generator(5, (40, 40), 1, 0.1)

	for i in range(10):
		chars, img, lm = gen.generate()
		img = Image.fromarray(img)
		img = img.resize( (120, 120) )
		# img.show()

		img.save("img/sample_" + str(i) +".gif")