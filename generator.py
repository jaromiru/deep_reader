from PIL import Image, ImageDraw, ImageFont
import random, string
import numpy as np

class Generator():
	def __init__(self, length, size, margin, noise):
		self.length = length
		self.fnt = ImageFont.truetype('arial.ttf', 8)
		self.size = size
		self.margin = np.array(margin)
		self.noise = noise

	def generate(self):
		img = Image.new('L', self.size)

		chars = [random.choice(string.ascii_uppercase) for i in range(self.length)]
		txt = "".join(chars)

		d = ImageDraw.Draw(img)

		txtsize = d.textsize(txt, font=self.fnt)
		pos = np.random.rand(2) * (self.size - 2 * self.margin - txtsize) + self.margin
		d.text(pos, txt, font=self.fnt, fill=128)
		 
		noise = self.noise * np.random.randn(*self.size) * 255
		img = img + noise
		# img = Image.fromarray(img)

		return chars, img

if __name__ == "__main__":
	gen = Generator(3, (30, 30), 1, 0.1)

	for i in range(10):
		chars, img = gen.generate()
		img = Image.fromarray(img)
		img = img.resize( (120, 120) )
		img.show()