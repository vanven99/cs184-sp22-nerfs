import OpenGL
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import pyglet
from PIL import Image
import numpy as np
def texture():
    #convert image to bytes
    png = Image.open("chair.png")
    bytes = np.asarray(png)

    #opengl context
    config = pyglet.gl.Config(alpha_size=8)

    #define gl ID
    ID = glGenTextures(1)
    #bind ID to byte array
    glBindTexture(GL_TEXTURE_2D, ID)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 2500, 2500, 0, GL_RGB,
    GL_UNSIGNED_BYTE, bytes)
    
    #tell renderer how to handle minification/magnification cases
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
if __name__ == "__main__":
    texture()
