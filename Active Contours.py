#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
from skimage import io
from skimage.color import rgb2gray
from skimage import io, filters
import matplotlib.pyplot as plt
import cv2


# In[7]:


image = io.imread('melanoma.JPG')
image_gray = rgb2gray(image)
n_points = 40
s = np.linspace(0, 2*np.pi, n_points)
x = np.round(100 + 95*np.cos(s))
y = np.round(75 + 70*np.sin(s))
snake  = np.array([x, y]).T
snake = snake.astype(int)
fig, ax = plt.subplots()
ax.imshow(image) 
ax.plot(snake[:, 0], snake[:, 1], '--r', lw=3)
ax.axis([0, image.shape[1], image.shape[0], 0])
plt.show()


# In[75]:


def cost_f(image_gray,points):
    alpha = 0.5  # magnitude of elasticity term
    beta = 1.0  # magnitude of curvature term
    gamma = 0.1
    gradient_x_b = np.zeros(image_gray.shape)
    gradient_y_b = np.zeros(image_gray.shape)
    smoothed_image = filters.gaussian(image_gray, sigma=2)
    for i in range(1,150):
        gradient_x_b[i,:] =  smoothed_image[i,:] - smoothed_image[i-1,:] 
        for j in range(1,200):
            gradient_y_b[:,j] =  smoothed_image[:,j] - smoothed_image[:,j-1]
    gradient_mag_b = -(np.square(gradient_x_b) + np.square(gradient_y_b))
    curvature = np.zeros((39))
    elasticity = np.zeros((39))
    curvature_tot = np.zeros((39))
    for i in range(39):
        curvature[i] = np.sum(snake[i+1][0] - snake[i][0])**2 + (snake[i+1][1] - snake[i][1])**2
        curvature_tot[i] = np.sum(curvature)
    elasticity[0] = 4.5
    for j in range(1,39):
        elasticity[j] = (snake[j+1][0]- 2*snake[j][0] + snake[j-1][0]) **2 + (snake[j+1][1]- 2*snake[j][1] + snake[j-1][1]) **2
    energy_internal = alpha * elasticity + beta * curvature
    for i in range(39):
        energy_ext = gradient_mag_b[snake[i][1],snake[i][0]]
    cost_f = energy_ext + gamma * energy_internal
    return cost_f


# In[78]:


gradient_x_b = np.zeros(image_gray.shape)
gradient_y_b = np.zeros(image_gray.shape)
smoothed_image = filters.gaussian(image_gray, sigma=2)
for i in range(1,150):
    gradient_x_b[i,:] =  smoothed_image[i,:] - smoothed_image[i-1,:] 
    for j in range(1,200):
        gradient_y_b[:,j] =  smoothed_image[:,j] - smoothed_image[:,j-1]
gradient_mag_b = -(np.square(gradient_x_b) + np.square(gradient_y_b))
plt.imshow(gradient_mag_b,cmap='gray')


# In[83]:


import numpy as np
def minimize_energy(contour, iterations,image_gray):
    dp_table = np.zeros((len(contour), len(contour)))
    for i in range(len(contour)):
        for j in range(i + 1, len(contour)):
            cost = cost_f(image_gray, contour)
            dp_table[i, j] = cost[i]
            dp_table[j, i] = cost[i]
    for _ in range(iterations):
        for i in range(1, len(contour) - 1):
            min_cost = np.inf
            for j in range(i + 1, len(contour) - 1):
                cost = dp_table[i, j] + dp_table[j, -1]
                if cost < min_cost:
                    min_cost = cost
                    min_idx = j
            dp_table[0, i] = min_cost
            dp_table[i, -1] = dp_table[min_idx, -1]
    optimal_contour = [contour[0]]
    i = 0
    while i < len(contour) - 1:
        j = np.argmin(dp_table[i, :])
        optimal_contour.append(contour[j])
        i = j
    optimal_contour.append(contour[-1])
    return optimal_contour


# In[85]:


iterations = 30
x = minimize_energy(snake,iterations,image_gray)


# In[ ]:




