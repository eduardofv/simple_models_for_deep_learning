###### **Watch on Github**

I've been learning a bit about deep neural networks on [Udacity's Deep Learning course](https://www.udacity.com/course/deep-learning--ud730) (which is mostly a [Tensorflow](https://www.tensorflow.org/) tutorial). The course uses [notMNIST](http://yaroslavvb.blogspot.mx/2011/09/notmnist-dataset.html) as a basic example to show how to train simple deep neural networks and let you achieve with relative ease an impressive 96% precision. In my way to the top of [Mount Stupid](http://www.smbc-comics.com/?id=2475), I wanted to get a better sense of what the NN is doing, so I took the examples of the first two chapters and modified it to test an idea: create some simple models, generate some training and test data sets from them, and see how the deep network performs and how different parameters affect the result. The models I'm using are two dimension (x,y) real values from [-1,1] that can be classified in two classes, either 0 or 1, inside or outside, true or false. The classification is a function that makes increasingly hard to classify the samples:

*   **positivos**: A trivial classification - if x>0 class=1 else class=0 [![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig1-positivos.png)](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig1-positivos.png)
*   **linear**: if y>x class=1 else class=0 [![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig2-linear.png)](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig2-linear.png)
*   **circle**: if (x,y) is inside a circle of radius=r then class=1 else class=0 [![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig3-circle.png)](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig3-circle.png)
*   **ring**: if (x,y) is inside a circle of radius=r but outside a concentric circle or radius=r/2 then class=1 else class=0 [![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig4-ring.png)](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig4-ring.png)
*   **cos**: if (x,y) is above a cosine with frequency=r then class=1 else class=0 [![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig5-cos.png)](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig5-cos.png)
*   **polar**: if (x,y) is below a cosine of frequency r in polar coordinates (inside a "rose of n-petals") then class = 1 else class = 0 [![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig6-polar.png)](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig6-polar.png)

The plots shown above are generated from the training sets by setting positive samples as dark blue and negative samples as light blue. Two neural networks are defined on the [code](https://github.com/eduardofv/simple_models_for_deep_learning/blob/master/models_for_deep_learning.ipynb), with a single hidden layer and with two hidden layers. Several parameters may be adjusted for each run:

*   model function to use
*   number or nodes of each layer,
*   batch size and number of steps for the stochastic gradient descent of the training phase,
*   beta for regularisation,
*   standard deviation for the initialisation of the variables,
*   and learning rate decay for the second NN

Other parameters are treated as globals such as the size of the training, validation and test sets.  When the NN function is called the network is trained and tested, and the predictions are added to the plot marking them on gray if the prediction was correct and on red if the prediction as incorrect in order to see where the classifier is missing it. You can call the NN function varying the parameters to test what happens. It's also included some iterative code to test many variants and generate a *lot* of plots. Check the code and comment what you find. Here are some interesting results I've got:

## Trivial model with a single hidden node and few training data

As expected, the model is too simple to really learn anything: it's basically classifying everything as negative and thus gets a accuracy of 49.9%[

![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig7-n1-pos-1-100-10.png)

](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig7-n1-pos-1-100-10.png) 

## Trivial model with a single hidden node with more training data

By increasing the size of the batches and steps for the stochastic gradient descent even a single node on the hidden layer yields good results: 99.6% accuracy[

![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig8-n1-pos-1-1000-100.png)

](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig8-n1-pos-1-1000-100.png) 

## Linear model with a single hidden node and few training data

Of course few training data does anything good for a slightly more complex model: 50.1% accuracy.[

![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig9-n1-lin-1-100-10.png)

](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig9-n1-lin-1-100-10.png) 

## Linear model with a single hidden node with more training data

Again the classifier performs way better: 99.9%[

![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig10-n1-lin-1-1000-100.png)

](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig10-n1-lin-1-1000-100.png) 

## Linear model with a 100 hidden nodes with few training data

Can hidden nodes compensate for few training samples? 100 nodes, batch of 100 and 10 steps does certainly better at 93.9% accuracy and an interesting plot with linear classification on a wrong (but close) slope.[

![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig11-n1-lin-100-100-10.png)

](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig11-n1-lin-100-100-10.png) 

## Linear model with a 1000 hidden nodes with few training data

What about 1000 hidden nodes and same parameters as before? It certainly gets you closer but not still on it: 98% accuracy. Watching the slope, one can think it may be overfitting.[

![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig12-n1-lin-1000-100-10.png)

](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig12-n1-lin-1000-100-10.png) 

## Linear model with **lots** of hidden nodes and few training data

First tried 10,000 hidden nodes (got 98.9%) and then 100,000 nodes (shown below) and accuracy jumped back to 98.5%, clearly overfitting. For this case, a wider network behaves better but just to a certain point.[

![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig13-n1-lin-100000-100-10.png)

](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig13-n1-lin-100000-100-10.png) 

## Circle with a previously good classifier

Training the classifier that previously behaved well with the circular model shows clearly it needs more data or nodes.[

![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig15-n1-circ-1-1000-100.png)

](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig15-n1-circ-1-1000-100.png) 

## Circle with a single node

Turns out that it seems a single node can't capture the complexity of the model. Varying the training parameters looks that the NN is always trying to adjust a linear classifier.

[![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig16-n1-circ-1-1000-6000.png)](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig16-n1-circ-1-1000-6000.png)[![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig17-n1-circ-1-10000-10000.png)](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig17-n1-circ-1-10000-10000.png)[![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig18-n1-circ-1-1000-100000.png)](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig18-n1-circ-1-1000-100000.png) 

## Circle with a many nodes and different training sizes

### Two nodes

With two nodes the model looks like fitting two linear classifiers. Even varying the training parameters results are similar, except in some cases with more training data that looks like the last examples of single node (maybe overfitting?)

[![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig19-n1-circ-2-100-1000.png)](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig19-n1-circ-2-100-1000.png)[![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig20-n1-circ-2-10000-10000.png)](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig20-n1-circ-2-10000-10000.png)

### Three and more nodes

Using three nodes things become interesting. For a batch size of 100 and 1000 steps the NN gives a pretty good approximation to the circle and goes beyond adjusting 3 linear classifiers. Something similar happens varying the training parameters.[

![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig21-n1-circ-3-100-1000.png)

](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig21-n1-circ-3-100-1000.png)Increasing the number of nodes increases the precision and the visual adjustment to the circle. Check for 10 (97.8%), 100 (99%) and seems to stop at some point 1000 (99%)  

[![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig22-n1-circ-10-100-1000.png)](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig22-n1-circ-10-100-1000.png)[![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig23-n1-circ-100-100-1000.png)](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig23-n1-circ-100-100-1000.png)[![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig24-n1-circ-1000-100-1000.png)](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig24-n1-circ-1000-100-1000.png)

Finally increasing the training parameters on the best classifier gives us a nice 99.5% accuracy, but not sure if it's overfitting[

![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig25-n1-circ-100-1000-10000.png)

](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig25-n1-circ-100-1000-10000.png) 

## The Ring

For the ring I started testing one of the best performers on the circle: 100 nodes, batch size of 100 and 1000 steps. Somehow expected, it tried to adjust to a single circle and missed the inner one.[

![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig26-n1-ring-100-100-1000.png)

](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig26-n1-ring-100-100-1000.png)Increasing the training parameters gave an interesting and unexpected result:[

![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/n1-ring-100-100-10000-ex1.png)

](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/n1-ring-100-100-10000-ex1.png)I also found that running many times with the same parameters may yield to different results. Both cases could be explained by the random initialisation and the stochastic gradient descent as the last example looks like a local minimum. Check below another interesting result using the exact same parameters yielding a pretty accurate classifier (92.7%)[

![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/n1-ring-100-100-10000-ex2.png)

](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/n1-ring-100-100-10000-ex2.png)Increasing the number of nodes and the training parameters improves accuracy and makes more likely to get a classifier with a pretty good level of accuracy (96.9%). Shown below, 1000 nodes, batch size of 1000 and 10000 steps.[

![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig27-n1-ring-1000-1000-10000.png)

](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig27-n1-ring-1000-1000-10000.png)

## Cosine

With few nodes and different training parameters we see the classifier struggling to fit.

[![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig28-n1-cos-1-1000-100.png)](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig28-n1-cos-1-1000-100.png)[![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig29-n1-cos-5-1000-100.png)](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig29-n1-cos-5-1000-100.png)[![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig30-n1-cos-10-100-1000.png)](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig30-n1-cos-10-100-1000.png)[![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig31-n1-cos-10-100-3000.png)](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig31-n1-cos-10-100-3000.png)

Increasing the parameters gives a model that doesn't increase very much the accuracy (around 87%). From the ring and the cosine it looks like there are limits to the complexity a single hidden layer can handle.[

![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig32-n1-cos-100-100-3000.png)

](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig32-n1-cos-100-100-3000.png)[

![](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig33-n1-cos-10000-1000-3000.png)

](https://raw.githubusercontent.com/eduardofv/simple_models_for_deep_learning/master/blog_imgs/fig33-n1-cos-10000-1000-3000.png)
