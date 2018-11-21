import argparse
import cv2
import os

OUTPUT_PATH = './output.mp4'
FPS = 6
HEIGHT = 256
WIDTH = 512

parser = argparse.ArgumentParser(description='Convert a series of images to a single video.')
parser.add_argument('img_list', type=str, help='List describing the path to images')
parser.add_argument('--fps', type=int, default=FPS, help='FPS of the output video')
parser.add_argument('--output_path', type=str, default=OUTPUT_PATH, help='Path to the output file')
parser.add_argument('--width', type=int, default=WIDTH, help='width of images')
parser.add_argument('--height', type=int, default=HEIGHT, help='height of images')

args = parser.parse_args()

def main():
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	video = cv2.VideoWriter(args.output_path, fourcc, args.fps, (args.width, args.height))
	#print (args.img_list)
	img_fname = [fn.strip() for fn in open(args.img_list, 'r')]
	#print (img_fname)
	for fname in img_fname:
		img = cv2.imread(fname)
		video.write(img)

	video.release()


if __name__ == '__main__':
	main()
