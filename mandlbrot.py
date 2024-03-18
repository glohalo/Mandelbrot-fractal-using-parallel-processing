#Mandelbrot fractal using parallel processing:
import numpy as np #numerical computations
from PIL import Image #for image creation and manipulation
from numba import jit, prange # for JIT (just-in-time) compilation to speed up the computation
@jit
def mandelbrot(c, threshold):
    """
    This function takes a complex number c and a threshold value
    - It iterates the Mandelbrot function up to a certain threshold to
    determine if the point c is within the Mandelbrot set
    - If the absolute value of z exceeds 2, it returns the numbers of 
    iterations it took to exceed the threshold. Otherwise, it returns the threshold value
    """
    z= 0
    for i in range(threshold):
        if abs(z)>2:
            return i
        #z = z*(1-z) + c #logistic function
        #gamma
        z = (z + 1)**2 + c
        #z= z*z+c
    return threshold

@jit(parallel=True)
def generate_mandelbrot(img, min_re, max_re, min_im, max_im, threshold):
    """
    This funtion generates the fractal image
    - It takes an image object img, minimum and maximum real values min_re 
    and max_re, minimum and maximum imaginary values min_im and max_im, and 
    the iteration threshold.
    - It iterates through each pixel of the image, calculates the corresponding 
    complex number c, and computes the Mandelbrot iteration using the mandelbrot 
    function.
    - It sets the pixel color based on the number of iterations it took for the
    point to scape the threshold
    """
    width, height = img.shape[1], img.shape[0]
    for x in prange(width):
        for y in range(height):
            cx = x * (max_re - min_re) / width + min_re
            cy = y * (max_im - min_im) / height + min_im
            c = complex(cx, cy)
            color_index = mandelbrot(c, threshold)
            #img[y, x] = (color_index, color_index, color_index) # all black and border white
            img[y, x] = map_color(color_index, threshold) #if you want with border blue and within the set with white
@jit
def map_color(iterations, threshold):
    """
    Maps the number of iterations to a color gradient.
    """
    if iterations == threshold:
        return (255, 255, 255)  # Set points within the Mandelbrot set to black
    else:
        # Example color gradient from blue to white
        blue_value = int(255 * iterations / threshold)
        return (0, 0, blue_value)
def main():
    """
    This function sets up the parameters for generating the Mandelbrot fractal image
    - It creates a new image objects, calls the "generate_mandelbrot"
    function to generate the fractals, saves the image and prints a message indicating that the image
    has been generated
    - he if __name__ == "__main__": block ensures that the main function is executed when the script 
    is run directly, but not when it's imported as a module into another script.
    """
    imgx, imgy = 1000 * 10, 800 * 10
    min_re, max_re = -2.5, 1.0
    min_im, max_im = -1.2, 1.2
    threshold = 1000

    img = np.zeros((imgy, imgx, 3), dtype=np.uint8)
    generate_mandelbrot(img, min_re, max_re, min_im, max_im, threshold)
    img = Image.fromarray(img)
    img.save("/Users/gloriacarrascal/fractals/images/mandelbrot_colorgamma.png")
    print("Generated mandelbrot_colorgamma.png")

if __name__ == "__main__":
    main()
