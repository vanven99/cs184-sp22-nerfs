# import OpenGL
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import pyglet
from PIL import Image
import numpy as np
def texture(filename, size):
    #convert image to bytes
    png = Image.open(filename)
    bytes = np.asarray(png)
    bytes = np.flipud(bytes)

    #opengl context
    config = pyglet.gl.Config()

    #define gl ID
    ID = glGenTextures(1)
    #bind ID to byte array
    glBindTexture(GL_TEXTURE_2D, ID)
    #print(bytes)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, size, size, 0, GL_RGB,
        GL_UNSIGNED_BYTE, bytes)

    #tell renderer how to handle minification/magnification cases
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    return ID
# if __name__ == "__main__":
#     texture()