# Tensorflow - Image Classification

Follow along as we use Tensorflow to demonstrate image classification of multiple classes. We will be using the Inception model, trained on academic benchmarks, to distinguish between 6 shapes, cube, sphere, rectangular prism, pyramid, cylinder, and cone.


- - -

## My system
```
Operating system:  MacOS Sierra
System version:    10.12.3 (16D32)
Processors:        2.7 GHz Intel Core i5
Memory:            8 GB 1867 Mhz DDR3
Graphics:          Intel Iris Graphics 6100 1536 MB
```
## Python and Docker versions
```
Python 2.7.14
Docker Client:
 Version:      18.03.1-ce
 API version:  1.37
 Go version:   go1.9.5
 Git commit:   9ee9f40
 Built:        Thu Apr 26 07:13:02 2018
 OS/Arch:      darwin/amd64
```
## Docker Quickstart Terminal: Installation & Check
We will use the Docker Quickstart terminal, that you can download from the [Docker toolbox](https://docs.docker.com/toolbox/toolbox_install_mac/#step-1-check-your-version). 

When you successfully download the Docker Quickstart Terminal, start it up and at the top there will be a neat whale.
If you type the `docker run hello-world` command into your terminal and press RETURN, you should get output similar to the following.
```

                        ##         .
                  ## ## ##        ==
               ## ## ## ## ##    ===
           /"""""""""""""""""\___/ ===
      ~~~ {~~ ~~~~ ~~~ ~~~~ ~~~ ~ /  ===- ~~~
           \______ o           __/
             \    \         __/
              \____\_______/


docker is configured to use the default machine with IP 192.168.99.100
For help getting started, check out the docs at https://docs.docker.com

RhoamRois:tf_files Home$ docker run hello-world

Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/engine/userguide/
```
*You can also check the docker version by typing the `docker version` command.*

## Docker Quickstart Terminal: Login
You'll want to login to be able to push and pull images from Docker Hub.<br>
This is important!
```
RhoamRois:tf_files Home$ docker login
Login with your Docker ID to push and pull images from Docker Hub. If you don't have a Docker ID, head over to https://hub.docker.com to create one.
Username (rdrois): 
Password: 
Login Succeeded
```

## Tensorflow Docker image setup
In the Docker Quickstart Terminal, you'll need to download the Tensorflow Docker image and enter a shell within it. 
1. Type `docker run -it tensorflow/tensorflow:latest-devel` and press RETURN.<br>
You will know it has succeeded when you automatically enter a shell within the image container. You'll also see each new line starting with `root@` followed by 12 alpha-numeric characters. 
2. Check to see if your tensorflow is updated.
```
RhoamRois:tf_files Home$ docker run -it tensorflow/tensorflow:latest-devel
root@13bae75b699f:~# cd /tensorflow/
root@13bae75b699f:/tensorflow# git pull
Already up-to-date.
```

To exit the container and return to your local terminal, type the `exit` command or press CTRL+d.

## Acquire repository to get the training shape images:
In a desired local directory, clone the repository to get the training shape images:
```
RhoamRois:tf_files Home$ git clone https://github.com/rdrois/basic-shapes
```
#### 3 major items in repository
* shapes: This directory holds 6 directories, each with images with their respective shape. The shapes are: cube, sphere, rectangular prism, pyramid, cylinder, and cone.
* retrain.py: This retrain script is used to create a model distinguishes between the shapes.
* label_image.py: This script will help the network label a target image after the network has been retrained.

Note 1: The `retrain.py` and `label_image.py` scripts were found at [TransferLearnColab](https://github.com/EN10/TransferLearnColab). They can each be downloaded by typing the following.
```
Home$ curl -LO https://github.com/tensorflow/hub/raw/master/examples/image_retraining/retrain.py
```
```
Home$ curl -LO https://github.com/tensorflow/tensorflow/raw/master/tensorflow/examples/label_image/label_image.py
```

## Run Docker container with image directory access
#### Map local to container
It is important that while in the Docker container, access to local files and folders is still possible. 
To do this, we need to use the `-v` argument in the docker command to allow us to map a directory locally to a directory inside our Docker container.
```
RhoamRois:tf_files Home$ docker run -it -v ~/Documents/tf_files/basic-shapes:/basic-shapes tensorflow/tensorflow:latest-devel
root@5c70540022ff:~# 
```

#### Check is map works
After you've mapped your directory to Docker, you can check if the container can list directory contents.
As of right now, you're in the tf_files directory, so using the `ls /tf_files/` command will not work.
Instead, try the `ls /basic-shapes/` command to show a list of the directory's contents.
```
root@5c70540022ff:~# ls /tf_files/
ls: cannot access '/tf_files/': No such file or directory
root@5c70540022ff:~# ls /basic-shapes/
LICENSE  label_image.py  retrain.py  shapes
LICENSE  README.md  ball.jpg  label_image.py  luxor.jpg  retrain.py  retrained_graph.pb  rubrix.jpg  shapes
root@5c70540022ff:~# 
```

#### Tensorflow_hub
The retraining script needs this so type the `pip install tensorflow_hub` command.

## Retraining script: Let's finally begin!
#### Run the script
We must be weary of which directory is being accessed for the retraining to work properly. 
```
root@5c70540022ff:~# python /basic-shapes/retrain.py --bottleneck_dir=/basic-shapes/bottlenecks --how_many_training_steps 500 --model_dir=/basic-shapes/inception --output_graph=/basic-shapes/retrained_graph.pb --output_labels=/basic-shapes/retrained_labels.txt --image_dir=/basic-shapes/shapes
```
While we wait for the retraining to finish, let's unpack the arguments of the command:
* `bottleneck_dir`: Since we are only training the last layer of our Inception network, which is called the bottleneck, we can safely cache the output of the network for every image up until the last layer, to speed up the training process. This cache is what's stored in `bottleneck_dir`.
* `how_many_training_steps`: Specifies the number of training iterations that our retraining process goes through. Since we have a small training set, we can keep this to a low value of `500`, but the recommended default is `4000`.
* `model_dir`: Specifies where the Inception network is stored.
* `output_graph`: Specifies where the specification of the network, i.e. the graph of operations that the network performs, is stored.
* `output_labels`: Specifies where the labels that our network can recognize, which in this case will be our 5 dog kinds (basset, bluetick, borzoi, chihuahua, redbone), will be stored.
* `image_dir`: Specifies the image directory where our training data is stored.

#### Check the outputs
As mentioned above, there are certain files that should have been created by runnint the script. 
The three items that should appear in the `basic-shapes` directory are:
* bottlenecks directory
* retrained_graph.ph
* retrained_labels.txt

If the script seemed to have run smoothly, yet these items did not appear, it is possible that there was directory mishap when executing the python command when running the script.

The output starts with looking for each category, then initialized InceptionV3.
The major actions taken by the script are:
* Searches for each category used for classification.
* Initializes InceptionV3 for each category.
* Creates bottlenecks cache files.
* Training accuracy, cross entropy, and validation accury are calculated over 500 steps.
* Reinitializes InceptionV3.
With the current setup, the script takes about 46 minutes.

```
INFO:tensorflow:Looking for images in 'cone'
INFO:tensorflow:Looking for images in 'cube'
INFO:tensorflow:Looking for images in 'cylinder'
INFO:tensorflow:Looking for images in 'pyramid'
INFO:tensorflow:Looking for images in 'rectangular_prism'
INFO:tensorflow:Looking for images in 'sphere'
INFO:tensorflow:Using /tmp/tfhub_modules to cache modules.
INFO:tensorflow:Downloading TF-Hub Module 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'.
INFO:tensorflow:Downloaded TF-Hub Module 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'.
...
INFO:tensorflow:Initialize variable module/InceptionV3/...
...
INFO:tensorflow:Creating bottleneck at /basic-shapes/bottlenecks/...
...
INFO:tensorflow:2018-07-20 06:53:15.987792: Step 0: Train accuracy = 43.0%
INFO:tensorflow:2018-07-20 06:53:15.989958: Step 0: Cross entropy = 1.735130
INFO:tensorflow:2018-07-20 06:53:17.013777: Step 0: Validation accuracy = 40.0% (N=100)
...
INFO:tensorflow:2018-07-20 06:55:44.356776: Step 499: Train accuracy = 75.0%
INFO:tensorflow:2018-07-20 06:55:44.358384: Step 499: Cross entropy = 0.755962
INFO:tensorflow:2018-07-20 06:55:44.620337: Step 499: Validation accuracy = 71.0% (N=100)
...
INFO:tensorflow:Initialize variable module/InceptionV3/...
...
INFO:tensorflow:Restoring parameters from /tmp/_retrain_checkpoint
INFO:tensorflow:Froze 378 variables.
INFO:tensorflow:Converted 378 variables to const ops.
```

## Label image with retrained model
This last part is about taking the retrained model and applying it to a new image.
The test images are already included in the `basic-shapes` directory.
Run the `label_image.py` script and the top 5 category fractions will be shown.

#### Results: ball.jpg
<p align="center">
<img src="https://github.com/RDRois/basic-shapes/blob/master/ball.jpg" width="300px" >
</p>

```
root@5c70540022ff:~# python /basic-shapes/label_image.py --graph=/basic-shapes/retrained_graph.pb --labels=/basic-shapes/retrained_labels.txt --output_layer=final_result --image=/basic-shapes/ball.jpg --input_layer=Placeholder
```
```
sphere 0.69433737
cube 0.1402661
cone 0.059887767
cylinder 0.045302395
pyramid 0.039126154
```
#### Results: luxor.jpg
<p align="center">
<img src="https://github.com/RDRois/basic-shapes/blob/master/luxor.jpg" width="300px" >
</p>

```
root@5c70540022ff:~# python /basic-shapes/label_image.py --graph=/basic-shapes/retrained_graph.pb --labels=/basic-shapes/retrained_labels.txt --output_layer=final_result --image=/basic-shapes/luxor.jpg --input_layer=Placeholder
```
```
pyramid 0.65736824
cone 0.15799783
cube 0.08429224
sphere 0.05765312
rectangular prism 0.02946632
```
#### Results: rubrix.jpg
<p align="center">
<img src="https://github.com/RDRois/basic-shapes/blob/master/rubrix.jpg" width="300px" >
</p>

```
root@a24f979cd6ab:/# python /basic-shapes/label_image.py --graph=/basic-shapes/retrained_graph.pb --labels=/basic-shapes/retrained_labels.txt --output_layer=final_result --image=/basic-shapes/rubrix.jpg --input_layer=Placeholder
```
```
cube 0.9530824
rectangular prism 0.016766412
cylinder 0.012242676
cone 0.011794928
sphere 0.003067277
```

## That's all folks!

- - -
- - -
#### Disclaimer
Through my journey in trying to get this image classifier to work, a combination of the following works was necessary.
[Tensorflow](https://www.tensorflow.org/hub/tutorials/image_retraining) image retraining tutorials.<br>
[Codelabs](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0) that classified flowers.<br>
[Siraj](https://www.youtube.com/watch?v=QfNvhPx5Px8&vl=en) that classified darth vader.<br>
[TransferLearnColab](https://github.com/EN10/TransferLearnColab) that classified flowers as well.<br>
[rhnvrm](https://github.com/rhnvrm/galaxy-image-classifier-tensorflow) that classified elliptical vs spiral galaxies.<br>

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

- - -
#### ¯\\_(ツ)_/¯
If you like this repository and would like to donate, <br>
please drop a nugget at the following Bitcoin address`3BVswsdQCAHcJW1k9syji4nuGHjwcAaaWt`
