import glfw
from OpenGL.GL import *
import numpy as np
import time

vertex_shader_source = """
#version 330 core
layout (location = 0) in vec2 aPos;
out vec2 TexCoords;
void main()
{
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoords = (aPos + 1.0) / 2.0;
}
"""

fragment_shader_source = """
#version 330 core
in vec2 TexCoords;
out vec4 FragColor;

uniform sampler2D uState;
uniform float uDu;
uniform float uDv;
uniform float uF;
uniform float uK;
uniform float uTimeStep;

void main()
{
    vec2 texelSize = 1.0 / textureSize(uState, 0);
    vec2 uv = TexCoords;

    vec2 laplacian = texture(uState, uv + texelSize * vec2(1, 0)).rg +
                     texture(uState, uv + texelSize * vec2(-1, 0)).rg +
                     texture(uState, uv + texelSize * vec2(0, 1)).rg +
                     texture(uState, uv + texelSize * vec2(0, -1)).rg -
                     4.0 * texture(uState, uv).rg;

    vec2 state = texture(uState, uv).rg;
    float dudt = uDu * laplacian.x - state.x * state.y * state.y + uF * (1.0 - state.x);
    float dvdt = uDv * laplacian.y + state.x * state.y * state.y - (uF + uK) * state.y;

    FragColor = vec4(state.x + dudt * uTimeStep, state.y + dvdt * uTimeStep, 0.0, 1.0);
}
"""

def main():
    if not glfw.init():
        return

    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
    window = glfw.create_window(800, 600, "Gray-Scott Reaction-Diffusion", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    vertex_shader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex_shader, vertex_shader_source)
    glCompileShader(vertex_shader)

    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragment_shader, fragment_shader_source)
    glCompileShader(fragment_shader)

    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)

    vertices = np.array([-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)

    width, height = 256, 256
    state0 = np.random.rand(width, height, 2).astype(np.float32)
    state0[:,:,1] = state0[:,:,0]
    state1 = np.copy(state0)

    texture_ids = glGenTextures(2)
    glBindTexture(GL_TEXTURE_2D, texture_ids[0])
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, width, height, 0, GL_RG, GL_FLOAT, state0)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    glBindTexture(GL_TEXTURE_2D, texture_ids[1])
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG32F, width, height, 0, GL_RG, GL_FLOAT, state1)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    current_state = 0  # Start with state0 as the current state

    glUseProgram(shader_program)
    glUniform1i(glGetUniformLocation(shader_program, "uState"), 0)
    glUniform1f(glGetUniformLocation(shader_program, "uDu"), 0.16)
    glUniform1f(glGetUniformLocation(shader_program, "uDv"), 0.08)
    glUniform1f(glGetUniformLocation(shader_program, "uF"), 0.035)
    glUniform1f(glGetUniformLocation(shader_program, "uK"), 0.065)

    last_time = time.time()

    while not glfw.window_should_close(window):
        current_time = time.time()
        time_step = current_time - last_time

        glUseProgram(shader_program)
        glUniform1f(glGetUniformLocation(shader_program, "uTimeStep"), time_step)
        
        # Set the texture based on the current state
        glUniform1i(glGetUniformLocation(shader_program, "uState"), 0)  # Use texture unit 0
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_ids[current_state])
        
        glBindVertexArray(VAO)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
        glBindVertexArray(0)
        
        glfw.swap_buffers(window)
        glfw.poll_events()

        # Swap current_state for double buffering
        current_state = 1 - current_state

        last_time = current_time

    glDeleteVertexArrays(1, [VAO])
    glDeleteBuffers(1, [VBO])
    glDeleteTextures(1, [texture_id])
    glDeleteProgram(shader_program)
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    glfw.terminate()

if __name__ == "__main__":
    main()
