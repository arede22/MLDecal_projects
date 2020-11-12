#!/usr/bin/env python
# coding: utf-8

# # Machine Learning at Berkeley: Machine Learning Decal
# ## Homework Three: Unsupervised Learning and Autoencoders
# 
# Release Date: February 27th, 2019
# Due Date: March 11th, 2019
# Contributing Authors: Brandon Trabucco
# 
# The goal of this homework is to familiarize you with various unsupervised learning and dimensionality reduction algorithms that are commonly used when handling large datasets. In particular, you will implement:
# 
# * Extracting The Dataset
# * Principal Component Analysis
# * A Linear Autoencoder
# * A Convolutional Autoencoder
# * (Optional) A Variational Autoencoder (VAE for short)
# 
# In addition to implementing these algorithms, you will use these algorithms to interpolate between existing data points, and extrapolate to new data points. Since images have nice visualizations, this homework shall use a miniature version of the CelebA (S. Yang et al. 2015) dataset that contains 5000 cropped images of celebrity faces. Feel free to download the full dataset after finishing the homework and tinkering with your models.
# 
# S. Yang, P. Luo, C. C. Loy, and X. Tang, "From Facial Parts Responses to Face Detection: A Deep Learning Approach", in IEEE International Conference on Computer Vision (ICCV), 2015

# In[1]:


get_ipython().run_cell_magic('capture', '', '# IMPORTANT: you must have all of these repositories properly installed on your machine to complete this homework.\n# you must also have ffmpeg installed. You may find the binaries at https://www.ffmpeg.org/download.html\n# Make sure you add the directories that contain the ffmpeg binaries to your path, reinstall matplotlib afterwards\nimport torch\nimport torchvision\nimport torch.nn.functional as F\nimport glob\nimport os\nfrom PIL import Image\nimport numpy as np\nimport matplotlib\nimport matplotlib.pyplot as plt\nimport matplotlib.animation as animation\nif not "ffmpeg" in matplotlib.animation.writers.list():\n    print("WARNING!!! You must add FFMPEG to your path before you can use the animations in this homework.")\nfrom IPython.display import HTML, display')


# ## Section One: Extracting The Dataset
# 
# In this section, you will extract the folder of images into a matrix $X \in \mathcal{R}^{N \times D}$ where the number of rows $N$ corresponds to the number of images in the dataset (5000 in total), and the number of features $D$ corresponds to the RGB values of every pixel in every image (32 * 32 * 3 = 3072 in this case).

# In[2]:


def extract_dataset(path_to_images, output_height, output_width, path_to_matrix):
    """Loads each image into memory, processes each image, and saves a matrix to the disk.
    Args:
        path_to_images: string, the path to the directory containing image files.
        output_height: integer, the height to scale each image to.
        output_width: integer, the width to scale each image to.
        path_to_matrix: string, the path where the matrix will be saved.
    """
    all_matching_files = glob.glob(os.path.join(path_to_images, "*.jpg"))
    X = np.zeros([len(all_matching_files), output_height * output_width * 3])
    for i, file in enumerate(all_matching_files):
        # TODO: fill in this section to accomplish the following.
        # 1) load the image with Image.open specified by its file path from the disk
        # 2) resize that image to be a [output_width, output_height] numpy array
        # 3) perform a row-major flatten of the array
        # 4) scale the elements of the array to be in the range [-1, 1]
        # 5) assign the array to the ith column of data matrix X
        # BEGIN YOUR CODE
        im = Image.open(file)
        arr = np.array(im.resize((output_width, output_height), resample=0))
        arr = arr.flatten()
        arr = np.interp(arr, (arr.min(), arr.max()), (-1, +1))
        X[i,:] = arr
        # END YOUR CODE
    np.save(os.path.join(path_to_matrix, "dataset.npy"), X)


# In[3]:


def load_dataset(path_to_matrix):
    """Loads a matrix containing processed images into the memory.
    Args:
        path_to_matrix: string, the path where the matrix was saved.
    Returns:
        a numpy matrix with 5000 rows (one per image) and 3072 columns.
    """
    return np.load(os.path.join(path_to_matrix, "dataset.npy"))


# In[4]:


def show_image(flat_image_vector, output_height, output_width):
    """Displays an image on jupyter notebook using matplotlib imshow.
    Args:
        flat_image_vector: a np.float32 vector with D = 3072 elements.
        output_height: integer, the height to reshape the image to.
        output_width: integer, the width to reshape the image to.
    """
    # TODO: fill in this section to accomplish the following.
    # 1) perform a row-major reshape from a flattened array to a [output_height, output_width, 3] tensor
    # 2) scale the elements of the array to be in the range [0, 1]
    # 3) render the image using matplotlib imshow(...)
    # 4) show() and close() the plot
    # BEGIN YOUR CODE
    flat_image_vector = flat_image_vector.reshape(output_height, output_width, 3)
    flat_image_vector = np.interp(flat_image_vector, (flat_image_vector.min(), flat_image_vector.max()), (0, +1))
    plt.imshow(flat_image_vector)
    plt.show()
    plt.close()
    # END YOUR CODE


# In[5]:


# TODO: fill in this section to accomplish the following.
# 1) call the extract_dataset function with the appropriate paths
# 2) assign the height 32 and the width 32
# BEGIN YOUR CODE
extract_dataset("C:/Users/anika/Downloads/celeba/celeba", 32, 32, "C:/Users/anika/Downloads")
# END YOUR CODE


# In[6]:


# TODO: fill in this section to accomplish the following.
# 1) call the load_dataset function with the appropriate path
# 2) assign the result to a data matrix named X
# BEGIN YOUR CODE
X = load_dataset(r"C:\Users\anika\Downloads")
# END YOUR CODE
print("A matrix with {0} images and {1} features per image was loaded.".format(*X.shape))


# In[7]:


# TODO: fill in this section to accomplish the following.
# 1) call the show_image function using a single row from the matrix X
# BEGIN YOUR CODE
show_image(X[0, :], 32, 32)
# END YOUR CODE


# ## Section Two: Principal Component Analysis
# 
# In this section, you will learn about Principal Component Analysis from an optimization perspective. You will then implement PCA to learn the $K$ principal components from the data matrix $X$. You will then use these principal components to interpolate between random rows of X. Finally, you will sample points in a lower dimensional subspace and invert PCA to generate new images of faces.

# The rows of $X$ lives in the space of $\mathcal{R}^{D}$. We define $D$ to be 3072 for the remainder of this homework. The principal components of $X$ provide a sequence of the best linear approximations to X in a lower dimensional subspace $\mathcal{R}^{Q}$ where the rank of the subspace $Q \leq D$ is no larger than the rank of the space that contains $X$. Consider a function of a vector $\lambda$ in $\mathcal{R}^{Q}$.
# 
# $$ f(\lambda) = \mu + V_{Q} \lambda $$
# 
# This function defines a linear transformation from the space of $\mathcal{R}^{Q}$ to the space of $\mathcal{R}^{D}$. There are two important parameters in this formulation: namely $\mu$ and $V_{Q}$. The vector $\mu$ is a position in the space of $\mathcal{R}^{D}$. The matrix $V_{Q} \in \mathcal{R}^{D \times Q}$ is a unitary matrix that maps the vector $\lambda$ from the subspace $\mathcal{R}^{Q}$ to the space of the data $\mathcal{R}^{D}$. The goal of PCA is to minimize the following reconstruction error.
# 
# $$ \min_{\mu, V_{Q}, \{ \lambda_{i}\} } \sum_{i = 1}^{N} || x_{i} - \mu - V_{Q} \lambda_{i} ||_{2}^{2} $$
# 
# Where the vector $x_{i}$ is the row in position i from the data matrix $X$, and the vector $\lambda_{i}$ represent the best approximation of the vector $x_{i}$ in the column space of the matrix $V_{Q}$. The other parameters have been previously defined, and are the same. We take this objective, and we optimize for $\mu$ and $\{ \lambda_{i}\}$.
# 
# $$ \mu = \frac{1}{N} \sum_{i = 1}^{N} x_{i} $$
# $$ \lambda_{i} = V_{Q}^{T} ( x_{i} - \mu ) $$
# 
# The optimization objective now amounts to solving for the optimal orthonormal matrix $V_{Q}$ that minimizes reconstruction error.
# 
# $$ \min_{V_{Q}} \sum_{i = 1}^{N} || x_{i} - \frac{1}{N} \sum_{i = 1}^{N} x_{i} - V_{Q} V_{Q}^{T} ( x_{i} - \frac{1}{N} \sum_{i = 1}^{N} x_{i} ) ||_{2}^{2} $$
# 
# The matrix resulting from $V_{Q} V_{Q}^{T}$ can be imagined a projection that maps each data point $x_{i}$ onto the best rank $Q$ approximation. See that we are subtracting the mean from each data point. If we assume each data point already has zero mean, the objective simplifies.
# 
# $$ \frac{1}{N} \sum_{i = 1}^{N} x_{i} = 0 \implies \min_{V_{Q}} \sum_{i = 1}^{N} || x_{i} - V_{Q} V_{Q}^{T} x_{i} ||_{2}^{2} $$
# 
# The solution may be obtained using Singular Value Decomposition. In particular, we can express the data matrix by its SVD $X = U \Sigma V^{T}$. Here, $U$ is an $N \times D$ orthogonal matrix. The matrix $U \Sigma$ represents the principal components of $X$, the directions with highest variance. The solution for $V_{Q}$ is simply to take the first $Q$ columns of the matrix $V$. This is left as an exercise for the reader and is not required.

# In[18]:


# TODO: fill in this section to accomplish the following.
# 1) using the function np.linalg.svd, calculate the singular value decomposition of the data matrix X
# 2) assign the SVD results to three matrices: U, S, V_T
# BEGIN YOUR CODE
A = np.linalg.svd(X, compute_uv=1)
U, S, V_T = A[0], A[1], A[2]
# END YOUR CODE


# In[19]:


# TODO: fill in this section to accomplish the following.
# 1) define Q = 256 to be the rank of the lower dimensional subspace in which you shall embed the data points
# 2) define variances to be the first Q singular values
# 3) define principal_components to be the basis vectors corresponding to the first Q singular values
# 4) define V_Q to be the matrix consisting of the first Q columns of V
# BEGIN YOUR CODE
Q = 256
variances = S[:Q]
principal_components = np.matmul(U[:, :Q], variances)
V_Q = np.transpose(V_T)[:, :Q]
# END YOUR CODE

print("The variances along the first {0} principal components are: {1}".format(Q, variances))
print("The first {0} principal components are: {1}".format(Q, principal_components))
print("The first {0} right singular vectors are: {1}".format(Q, V_Q))


# In[20]:


# TODO: fill in this section to accomplish the following.
# 1) select a single row of the data matrix X
# 2) project that row onto the rank-Q lower dimension subspace in R^D specified by the projection matrix (V_Q V_Q^T)
# BEGIN YOUR CODE
A = np.matmul(np.matmul(V_Q, np.transpose(V_Q)), X[1, :])
# END YOUR CODE


# In[21]:


# TODO: fill in this section to accomplish the following.
# 1) display the original image using show_image with height 32 and width 32
# 1) display the projected image using show_image with height 32 and width 32
# BEGIN YOUR CODE
show_image(X[1, :], 32, 32)
show_image(A, 32, 32)
# END YOUR CODE


# ### Comment on how well the best Q principal components reconstruct the image:
# The best Q principal components for me did not reconstruct the image very well.

# In[22]:


def latent_interpolation(z_one, z_two, reconstruction_function, output_height, output_width):
    """This function draws an interpolating animation from one image to another image in the latent space.
    Args:
        z_one: an np.float32 vector with Q elements.
        z_two: an np.float32 vector with Q elements.
        reconstruction_function: a function that takes in z_one or z_two and returns an np.float32 vector with D elements.
        output_height: integer, the height to reshape each image to.
        output_width: integer, the width to reshape each image to.
    """
    fig = plt.figure()
    im = None
    im = plt.imshow(reconstruction_function(z_one).reshape([output_height, output_width, 3]) / 2.0 + 0.5, animated=True)
    def updatefig(t):
        alpha = (0.5 * np.cos(t) + 0.5)
        im.set_array(reconstruction_function(z_one * alpha + z_two * (1.0 - alpha)).reshape([output_height, output_width, 3]) / 2.0 + 0.5)
        return im,
    display(HTML(animation.FuncAnimation(fig, updatefig, frames=np.linspace(0, 2*np.pi, 64), blit=True).to_html5_video()))
    plt.close()


# In[23]:


# TODO: fill in this section to accomplish the following.
# 1) select two different rows from the data matrix X
# 2) project each row onto the best Q principal components using V_Q^T
# 3) define reconstruction_function that reconstructs a data point x from its projection onto the best Q principal components using V_Q
# 4) call the function latent_interpolation and generate a visualization with height 32 and width 32 
# BEGIN YOUR CODE
z_one = np.matmul(np.transpose(V_Q), X[0, :])
z_two = np.matmul(np.transpose(V_Q), X[1, :])
def reconstruction_function(x):
    return np.matmul(V_Q, x)
latent_interpolation(z_one, z_two, reconstruction_function, 32, 32)
# END YOUR CODE


# ### Comment on what happens during the interpolation process:
# Interpolation process finds points in common between the two images and the animation plays through formulating a sequence whereby the image changes to find points in-common between the two photos and combines them into the original image.

# In[14]:


def latent_generation(z_mean, z_stddev, reconstruction_function, output_height, output_width):
    """This function samples from the latent space of the model and shows the resulting image.
    Args:
        z_mean: an np.float32 vector with Q elements.
        z_stddev: an np.float32 matrix with Q by Q elements.
        reconstruction_function: a function that takes in z_one or z_two and returns an np.float32 vector with D elements.
        output_height: integer, the height to reshape each image to.
        output_width: integer, the width to reshape each image to.
    """
    sampled_point = z_mean + z_stddev.dot(np.random.normal(0, 1, z_mean.shape))
    show_image(reconstruction_function(sampled_point), output_height, output_width)


# In[15]:


# TODO: fill in this section to accomplish the following.
# 1) project the data matrix onto best Q principal components using V_Q^T
# 2) compute z_mean as the average of the projected matrix along the 0th axis
# 3) define z_stddev to be the rank Q identity matrix for now
# 4) generate an new image by calling latent_generation with height 32 and width 32 
# BEGIN YOUR CODE
A = np.matmul(X, V_Q)
z_mean = np.mean(A, axis=0)
z_stddev = np.eye(Q)
latent_generation(z_mean, z_stddev, reconstruction_function, 32, 32)
# END YOUR CODE


# ### Comment on how real the generated face looks:
# The generated face does not look very real. It looks very pixelated and varies too much in color to be accurate. (And it looks pretty creepy.)

# ## Section Three: Linear Autoencoder
# 
# In this section, you will learn about the linear autoencoder, and you will also implement the linear autoencoder using pytorch. We shall use your linear autoencoder to interpolate between data points from $X$ and to also generate new samples of faces.

# In[16]:


class LinearEncoder(torch.nn.Module):
    
    def __init__(self, image_height, image_width, hidden_size):
        """Creates a single layer neural network.
        Args:
            image_height: an integer, the height of each image
            image_width: an integer, the width of each image
            hidden_size: an integer, the number of neurons in the hidden layer of this network
        """
        super(LinearEncoder, self).__init__()
        # TODO: fill in this section to accomplish the following.
        # 1) create a single layer neural network that performs a linear transformation from a vector with 
        #    image_height * image_width * 3 dimensions to a vector with hidden_size dimensions.
        #    HINT: consider the class torch.nn.Linear
        # BEGIN YOUR CODE
        self.singleLayer = torch.nn.Linear(image_height * image_width * 3, hidden_size, bias=True)
        # END YOUR CODE
        
    def forward(self, x):
        """Computes a single forward pass of this network.
        Args:
            x: a float32 tensor with shape [batch_size, D]
        Returns:
            a float32 tensor with shape [batch_size, hidden_size]
        """
        # TODO: fill in this section to accomplish the following.
        # 1) perform a forward pass using the hidden layer you defined
        # 2) return the resulting vector
        # BEGIN YOUR CODE
        return self.singleLayer(x)
        # END YOUR CODE


# In[17]:


class LinearDecoder(torch.nn.Module):
    
    def __init__(self, image_height, image_width, hidden_size):
        """Creates a single layer neural network.
        Args:
            image_height: an integer, the height of each image
            image_width: an integer, the width of each image
            hidden_size: an integer, the number of neurons in the hidden layer of this network
        """
        super(LinearDecoder, self).__init__()
        # TODO: fill in this section to accomplish the following.
        # 1) create a single layer neural network that performs a linear transformation from a vector with 
        #    hidden_size dimensions to a vector with image_height * image_width * 3 dimensions.
        #    HINT: consider the class torch.nn.Linear
        # BEGIN YOUR CODE
        self.singleLayer = torch.nn.Linear(hidden_size, image_height * image_width * 3, bias=True)
        # END YOUR CODE
        
    def forward(self, x):
        """Computes a single forward pass of this network.
        Args:
            x: a float32 tensor with shape [batch_size, hidden_size]
        Returns:
            a float32 tensor with shape [batch_size, D]
        """
        # TODO: fill in this section to accomplish the following.
        # 1) perform a forward pass using the hidden layer you defined
        # 2) return the resulting vector
        # BEGIN YOUR CODE
        return self.singleLayer(x)
        # END YOUR CODE


# In[18]:


# TODO: fill in this section to accomplish the following.
# 1) create an instance of LinearEncoder named linear_encoder with height 32, width 32, and hidden size Q
# 1) create an instance of LinearDecoder named linear_decoder with height 32, width 32, and hidden size Q
# 2) assign linear_autoencoder_loss to be an instance of torch.nn.MSELoss
# 2) create an optimizer named linear_autoencoder_optimizer of your choosing with a learning rate of your choosing.
#    HINT: consider the torch.optim.Adam object
# BEGIN YOUR CODE
linear_encoder = LinearEncoder(32, 32, Q)
linear_decoder = LinearDecoder(32, 32, Q)
linear_autoencoder_loss = torch.nn.MSELoss()
linear_autoencoder_optimizer = torch.optim.Adam(linear_encoder.parameters(), lr=0.001)
linear_autoencoder_optimizer = torch.optim.Adam(linear_decoder.parameters(), lr=0.001)
# END YOUR CODE


# In[19]:


# TODO: run the following section of code in order to train the model
# Construct a tensor from the dataset
image_tensor = torch.FloatTensor(X)
for i in range(1000):
    # Clear the previous gradient from the optimizer by calling .zero_grad()
    linear_autoencoder_optimizer.zero_grad()
    # Compute a full encoding and decoding step
    reconstructed_image = linear_decoder(linear_encoder(image_tensor))
    # Compute the mean squared reconstruction loss
    loss = linear_autoencoder_loss(reconstructed_image, image_tensor)
    # Pass the loss backward throgh the network, and compute the gradients
    loss.backward()
    # Update the optimizer by calling .step()
    linear_autoencoder_optimizer.step()
    # Return a detatched value of the loss for logging purposes
    print("On iteration {0} the loss was {1}.".format(i, loss.detach()))


# In[20]:


# TODO: run the following section of code to define the reconstruction function
def linear_autoencoder_reconstruction_function(z):
    return linear_decoder(torch.FloatTensor(z[np.newaxis, :])).detach()[0, :]


# In[21]:


# TODO: fill in this section to accomplish the following.
# 1) select two different rows from the data matrix X
# 2) compute the hidden representation for each row using your linear_encoder
# 3) call the function latent_interpolation and generate a visualization with height 32 and width 32 
# BEGIN YOUR CODE
z_one = linear_encoder(torch.FloatTensor(X[0, :]))
z_two = linear_encoder(torch.FloatTensor(X[1, :]))
latent_interpolation(z_one, z_two, linear_autoencoder_reconstruction_function, 32, 32)
# END YOUR CODE


# ### Comment on what happens during the interpolation process:
# This interpolation allows much more crossover for the two images than the previous interpolation process and can be seen in how the images morph into one another.

# In[23]:


# TODO: fill in this section to accomplish the following.
# 1) compute the hidden representation of each row of the data matrix
# 2) compute z_mean as the average of the hidden representation matrix along the 0th axis
# 3) define z_stddev to be the rank Q identity matrix for now
# 4) generate an new image by calling latent_generation with height 32 and width 32 
# BEGIN YOUR CODE
for i in range(256):
    A[i, :] = linear_encoder(torch.FloatTensor(X[i, :])).detach().numpy()
z_mean = np.mean(A, axis=0)
z_stddev = np.eye(Q)
latent_generation(z_mean, z_stddev, linear_autoencoder_reconstruction_function, 32, 32)
# END YOUR CODE


# ### Comment on how real the generated face looks:
# The generated face looks similar to the results rendered from PCA, but the images blend more together in this generated face. 

# ### Comment on how the linear autoencoder compares to PCA:
# The effect is similar to that of PCA, so I looked into the background of linear autoencoders and saw that linear autocoders are the same as PCAs. 

# ## Section Four: Convolutional Autoencoder
# 
# In this section, you will learn about the convolutional autoencoder, and you will also implement the convolutional autoencoder using pytorch. We shall use your convolutional autoencoder to interpolate between data points from $X$ and to also generate new samples of faces.

# In[7]:


class ConvolutionalEncoder(torch.nn.Module):
    
    def __init__(self, image_height, image_width, final_size):
        """Creates a deep convolutional neural network.
        Args:
            image_height: an integer, the height of each image
            image_width: an integer, the width of each image
            final_size: an integer, the depth of the final layer of this network
        """
        super(ConvolutionalEncoder, self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.final_size = max(16 * 3, final_size)
        # TODO: fill in this section to accomplish the following.
        # 1) create a 5 layer convolutional neural network that transforms an image with shape
        #    [image_height, image_width, 3] to a vector with final_size dimensions.
        #    HINT: consider the class torch.nn.Conv2d with stride=2
        # BEGIN YOUR CODE
        short = self.image_height * self.image_width
        self.layer1 = torch.nn.Conv2d(in_channels=3, out_channels=short * 2, kernel_size=2, stride=2, padding=1)
        self.layer2 = torch.nn.Conv2d(in_channels=short * 2, out_channels=short * 1, kernel_size=2, stride=2, padding=1)
        self.layer3 = torch.nn.Conv2d(in_channels=short * 1, out_channels=self.final_size * 3, kernel_size=2, stride=2, padding=1)
        self.layer4 = torch.nn.Conv2d(in_channels=self.final_size * 3, out_channels=self.final_size * 2, kernel_size=2, stride=2, padding=1)
        self.layer5 = torch.nn.Conv2d(in_channels=self.final_size * 2, out_channels=self.final_size, kernel_size=2, stride=2, padding=1)
        self.Sigmoid = torch.nn.Sigmoid()
        # END YOUR CODE
        
    def forward(self, x):
        """Computes a single forward pass of this network.
        Args:
            x: a float32 tensor with shape [batch_size, D]
        Returns:
            a float32 tensor with shape [batch_size, final_size]
        """
        x = x.view(x.size()[0], self.image_height, self.image_width, 3)
        x = torch.transpose(x, 1, 3)
        # TODO: fill in this section to accomplish the following.
        # 1) perform a forward pass using the conv layers you defined
        # 2) apply any activation function you want
        # BEGIN YOUR CODE
        x = self.Sigmoid(self.layer1(x))
        x = self.Sigmoid(self.layer2(x))
        x = self.Sigmoid(self.layer3(x))
        x = self.Sigmoid(self.layer4(x))
        x = self.Sigmoid(self.layer5(x))
        # END YOUR CODE
        x = torch.transpose(x, 1, 3)
        x = x.contiguous()
        x = x.view(x.size()[0], self.final_size)
        return x


# In[8]:


class ConvolutionalDecoder(torch.nn.Module):
    
    def __init__(self, image_height, image_width, final_size):
        """Creates a deep convolutional neural network.
        Args:
            image_height: an integer, the height of each image
            image_width: an integer, the width of each image
            final_size: an integer, the depth of the final layer of this network
        """
        super(ConvolutionalDecoder, self).__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.final_size = max(16 * 3, final_size)
        # TODO: fill in this section to accomplish the following.
        # 1) create a 5 layer transpose convolutional neural network that transforms a vector with 
        #    final_size dimensions to an image with shape [image_height, image_width, 3]
        #    HINT: consider the class torch.nn.ConvTranspose2d with stride=2
        # BEGIN YOUR CODE
        short = self.image_height * self.image_width
        self.layer1 = torch.nn.Conv2d(in_channels=self.final_size, out_channels=self.final_size * 2, kernel_size=2, stride=2, padding=1)
        self.layer2 = torch.nn.Conv2d(in_channels=self.final_size * 2, out_channels=self.final_size * 3, kernel_size=2, stride=2, padding=1)
        self.layer3 = torch.nn.Conv2d(in_channels=self.final_size * 3, out_channels=short * 1, kernel_size=2, stride=2, padding=1)
        self.layer4 = torch.nn.Conv2d(in_channels=short * 1, out_channels=short * 2, kernel_size=2, stride=2, padding=1)
        self.layer5 = torch.nn.Conv2d(in_channels=short * 2, out_channels=3, kernel_size=2, stride=2, padding=1)
        self.Sigmoid = torch.nn.Sigmoid()
        # END YOUR CODE
        
    def forward(self, x):
        """Computes a single forward pass of this network.
        Args:
            x: a float32 tensor with shape [batch_size, final_size]
        Returns:
            a float32 tensor with shape [batch_size, D]
        """
        x = x.view(x.size()[0], 1, 1, self.final_size)
        x = torch.transpose(x, 1, 3)
        # TODO: fill in this section to accomplish the following.
        # 1) perform a forward pass using the conv layers you defined
        # 2) apply any activation function you want
        # BEGIN YOUR CODE
        x = self.Sigmoid(self.layer1(x))
        x = self.Sigmoid(self.layer2(x))
        x = self.Sigmoid(self.layer3(x))
        x = self.Sigmoid(self.layer4(x))
        x = self.Sigmoid(self.layer5(x))
        # END YOUR CODE
        x = torch.transpose(x, 1, 3)
        x = x.contiguous()
        x = x.view(x.size()[0], self.image_height * self.image_width * 3)
        return x


# In[9]:


# TODO: fill in this section to accomplish the following.
# 1) create an instance of ConvolutionalEncoder named convolutional_encoder with height 32, width 32, and hidden size Q
# 1) create an instance of ConvolutionalDecoder named convolutional_decoder with height 32, width 32, and hidden size Q
# 2) assign convolutional_autoencoder_loss to be an instance of torch.nn.MSELoss
# 2) create an optimizer named convolutional_autoencoder_optimizer of your choosing with a learning rate of your choosing.
#    HINT: consider the torch.optim.Adam object
# BEGIN YOUR CODE
Q = 256
convolutional_encoder = ConvolutionalEncoder(32, 32, Q)
convolutional_decoder = ConvolutionalDecoder(32, 32, Q)
convolutional_autoencoder_loss = torch.nn.MSELoss()
convolutional_autoencoder_optimizer = torch.optim.Adam(convolutional_encoder.parameters(), lr=0.001)
convolutional_autoencoder_optimizer = torch.optim.Adam(convolutional_decoder.parameters(), lr=0.001)
# END YOUR CODE


# In[10]:


# TODO: run the following section of code in order to train the model
# Construct a tensor from the dataset
image_tensor = torch.FloatTensor(X)
for i in range(1000):
    # Clear the previous gradient from the optimizer by calling .zero_grad()
    convolutional_autoencoder_optimizer.zero_grad()
    # Compute a full encoding and decoding step
    reconstructed_image = convolutional_decoder(convolutional_encoder(image_tensor))
    # Compute the mean squared reconstruction loss
    loss = convolutional_autoencoder_loss(reconstructed_image, image_tensor)
    # Pass the loss backward throgh the network, and compute the gradients
    loss.backward()
    # Update the optimizer by calling .step()
    convolutional_autoencoder_optimizer.step()
    # Return a detatched value of the loss for logging purposes
    print("On iteration {0} the loss was {1}.".format(i, loss.detach()))


# In[31]:


# TODO: run the following section of code to define the reconstruction function
def convolutional_autoencoder_reconstruction_function(z):
    return np.asarray(convolutional_decoder(torch.FloatTensor(z[np.newaxis, :])).detach()[0, :], np.float32)


# In[121]:


# TODO: fill in this section to accomplish the following.
# 1) select two different rows from the data matrix X
# 2) compute the hidden representation for each row using your convolutional_encoder
# 3) call the function latent_interpolation and generate a visualization with height 32 and width 32 
# BEGIN YOUR CODE
z_one = np.asarray(convolutional_encoder(torch.FloatTensor(X[0, :])).detach(), np.float32)
z_two = np.asarray(convolutional_encoder(torch.FloatTensor(X[1, :])).detach(), np.float32)
latent_interpolation(z_one, z_two, convolutional_autoencoder_reconstruction_function, 32, 32)
# END YOUR CODE


# ### Comment on what happens during the interpolation process:
# [TODO: your response here]

# In[123]:


# TODO: fill in this section to accomplish the following.
# 1) compute the hidden representation of each row of the data matrix using convolutional_encoder
# 2) compute z_mean as the average of the hidden representation matrix along the 0th axis
# 3) define z_stddev to be the rank Q identity matrix for now
# 4) generate an new image by calling latent_generation with height 32 and width 32 
# BEGIN YOUR CODE
for i in range(3072):
    A[i, :] = convolutional_encoder(X[i, :])
z_mean = np.mean(A, axis=0)
z_stddev = np.eye(Q)
latent_generation(z_mean, z_stddev, convolutional_autoencoder_reconstruction_function, 32, 32)
# END YOUR CODE


# ### Comment on how real the generated face looks:
# [TODO: your response here]

# ### Comment on how the convolutional autoencoder compares to the linear autoencoder and PCA:
# [TODO: your response here]

# ## (Optional) Section Five: Variational Autoencoder
# 
# In this section, we implement the Variational Autoencoder, an extension for the traditional autoencoder that explicitly models the probability distribution of a latent variable. This section is optional, and so we fill in the code for you. If you have extra time after completing the rest of this homework, you should first read this tutorial on variational inference https://arxiv.org/pdf/1606.05908.pdf. Then, you may attempt to train the VAE given below.

# In[41]:


# TODO: run the following section of code
class Sampler(torch.nn.Module):
    
    def __init__(self, hidden_size):
        """Creates a Variational sampling layer.
        Args:
            hidden_size: an integer, the number of neurons in the sampling layer.
        """
        super(Sampler, self).__init__()
        self.hidden_size = hidden_size
        self.log_scale = torch.nn.Linear(hidden_size, hidden_size)
        self.shift = torch.nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        """Computes a single forward pass of this network.
        Args:
            x: a float32 tensor with shape [batch_size, hidden_size]
        Returns:
            a float32 tensor with shape [batch_size, hidden_size]
        """
        scale = torch.exp(self.log_scale(x))
        shift = self.shift(x)
        sample = torch.randn([self.hidden_size]) * scale + shift
        return sample
        
    def kl_penalty(self, x):
        """Computes a single forward pass of this network.
        Args:
            x: a float32 tensor with shape [batch_size, hidden_size]
        Returns:
            a float32 scalar: KL divergence between this distribution and the standard normal distribution.
        """
        log_scale = self.log_scale(x)
        scale = torch.exp(log_scale)
        shift = self.shift(x)
        return torch.mean(log_scale + (1.0 + shift * shift) / (2.0 * scale * scale) - 0.5)


# In[42]:


# TODO: run the following section of code
variational_encoder = ConvolutionalEncoder(32, 32, 256)
variational_decoder = ConvolutionalDecoder(32, 32, 256)
sampler = Sampler(256)
variational_autoencoder_loss = torch.nn.MSELoss()
variational_autoencoder_optimizer = torch.optim.Adam([
    {"params": variational_encoder.parameters()}, 
    {"params": variational_decoder.parameters()}, 
    {"params": sampler.parameters()}])


# In[43]:


# TODO: run the following section of code in order to train the model
# Construct a tensor from the dataset
image_tensor = torch.FloatTensor(X)
for i in range(10000):
    # Clear the previous gradient from the optimizer by calling .zero_grad()
    variational_autoencoder_optimizer.zero_grad()
    # Compute a full encoding and decoding step
    hidden_variables = variational_encoder(image_tensor)
    reconstructed_image = variational_decoder(sampler(hidden_variables))
    # Compute the mean squared reconstruction loss
    loss = variational_autoencoder_loss(reconstructed_image, image_tensor) - sampler.kl_penalty(hidden_variables)
    # Pass the loss backward throgh the network, and compute the gradients
    loss.backward()
    # Update the optimizer by calling .step()
    variational_autoencoder_optimizer.step()
    # Return a detatched value of the loss for logging purposes
    print("On iteration {0} the loss was {1}.".format(i, loss.detach()))


# In[44]:


# TODO: run the following section of code
def variational_autoencoder_reconstruction_function(z):
    return np.asarray(variational_decoder(torch.FloatTensor(z[np.newaxis, :])).detach()[0, :], np.float32)


# In[45]:


# TODO: run the following section of code
x_one = X[0, :]
x_two = X[1, :]
z_one = np.asarray(sampler(variational_encoder(torch.FloatTensor(x_one[np.newaxis, :]))).detach(), np.float32)
z_two = np.asarray(sampler(variational_encoder(torch.FloatTensor(x_two[np.newaxis, :]))).detach(), np.float32)
latent_interpolation(z_one, z_two, convolutional_autoencoder_reconstruction_function, 32, 32)


# ### Comment on what happens during the interpolation process:
# [TODO: your response here]

# In[46]:


# TODO: run the following section of code
z_mean = np.asarray(torch.mean(sampler(variational_encoder(torch.FloatTensor(X))), 0).detach(), np.float32)
z_stddev = np.identity(Q)
latent_generation(z_mean, z_stddev, convolutional_autoencoder_reconstruction_function, 32, 32)


# ### Comment on how real the generated face looks:
# [TODO: your response here]

# ### Comment on how the convolutional autoencoder compares to the linear autoencoder and PCA:
# [TODO: your response here]

# In[ ]:




