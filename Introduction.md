# ML Project Report - Introduction

## Introduction

Development project in Machine Learning is aimed at developing good programming practices, using standard development tools and getting used to collaborative work. 

The objective of the project is to apply a Machine Learning model onto two different datasets:

1. [Banknote Authentication Dataset](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)
2. [Chronic Kidney Disease](https://www.kaggle.com/datasets/mansoordaku/ckdisease)

### Machine Learning Workflow

1. Import the dataset
2. Clean the data, perform pre-processing
   * Replace missing values by average or median values
   * Center and normalize the data
3. Split the dataset
   * Split between training set and test set
   * Split the training set for cross-validation
4. Train the model (including feature selection)
5. Validate the model

### Datasets

#### [Banknote Authentication Dataset](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)

Number of Attributes: 4 + `class` = 5 (4 numeric, 1 nominal) 

Number of Instances:  1372

Attribute Information :

|      |                Meaning                |   Type    |
| :--: | :-----------------------------------: | :-------: |
|  1   | Variance of Wavelet Transformed image | Numerical |
|  2   | Skewness of Wavelet Transformed image | Numerical |
|  3   | Curtosis of Wavelet Transformed image | Numerical |
|  4   |           Entropy of image            | Numerical |
|  5   |                 Class                 |  Nominal  |

No missing or tainted value.

#### [Chronic Kidney Disease](https://www.kaggle.com/datasets/mansoordaku/ckdisease)

Number of Attributes: 24 + class = 25 (11 numeric, 14 nominal) 

Number of Instances:  400 (250 CKD, 150 notckd)

Attribute Information :

|      | Name  |         Meaning         |   Type    |              Unit               |
| :--: | :---: | :---------------------: | :-------: | :-----------------------------: |
|  1   |  age  |      Age in years       | Numerical |              Year               |
|  2   |  bp   |     Blood pressure      | Numerical |              mm/Hg              |
|  3   |  sg   |    Specific Gravity     |  Nominal  | (1.005,1.010,1.015,1.020,1.025) |
|  4   |  al   |         Albumin         |  Nominal  |          (0,1,2,3,4,5)          |
|  5   |  su   |          Sugar          |  Nominal  |          (0,1,2,3,4,5)          |
|  6   |  rbc  |     Red Blood Cells     |  Nominal  |        (normal,abnormal)        |
|  7   |  pc   |        Pus Cell         |  Nominal  |        (normal,abnormal)        |
|  8   |  pcc  |     Pus Cell clumps     |  Nominal  |      (present,notpresent)       |
|  9   |  ba   |        Bacteria         |  Nominal  |      (present,notpresent)       |
|  10  |  bgr  |  Blood Glucose Random   | Numerical |             mgs/dl              |
|  11  |  bu   |       Blood Urea        | Numerical |             mgs/dl              |
|  12  |  sc   |    Serum Creatinine     | Numerical |             mgs/dl              |
|  13  |  sod  |         Sodium          | Numerical |              mEq/L              |
|  14  |  pot  |        Potassium        | Numerical |              mEq/L              |
|  15  | hemo  |       Hemoglobin        | Numerical |               gms               |
|  16  |  pcv  |   Packed Cell Volume    | Numerical |                                 |
|  17  |  wc   | White Blood Cell Count  | Numerical |           cells/cumm            |
|  18  |  rc   |  Red Blood Cell Count   | Numerical |          millions/cmm           |
|  19  |  htn  |      Hypertension       |  Nominal  |            (yes,no)             |
|  20  |  dm   |    Diabetes Mellitus    |  Nominal  |            (yes,no)             |
|  21  |  cad  | Coronary Artery Disease |  Nominal  |            (yes,no)             |
|  22  | appet |        Appetite         |  Nominal  |           (good,poor)           |
|  23  |  pe   |       Pedal Edema       |  Nominal  |            (yes,no)             |
|  24  |  ane  |         Anemia          |  Nominal  |            (yes,no)             |
|  25  | class |          Class          |  Nominal  |          (ckd,notckd)           |

Tainted Attribute Values:

|      | Original Value | Correct Value |
| :--: | :------------: | :-----------: |
|  1   |      \t43      |      43       |
|  2   |     \t6200     |     6200      |
|  3   |     \t8400     |     8400      |
|  4   |      \tno      |      no       |
|  5   |     \tyes      |      yes      |
|  6   |      rbc       |      rbc      |
|  7   |     ckd\t      |      ckd      |
|  8   |      \t?       |               |

### Models

#### SVM

Support vector machine (SVM) are supervised learning models and associated learning algorithms for analyzing data in classification and regression analysis. Given a set of training instances, each labeled as belonging to one or the other of two classes, the SVM training algorithm creates a model that assigns new instances to one of the two classes, making it a non-probabilistic binary linear classifier. the SVM model represents instances as points in space, so that the mapping allows instances of separate classes to be separated by as wide an apparent interval as possible. The new instances are then mapped into the same space and the categories to which they belong are predicted based on which side of the interval they fall.

When the training data is linearly divisible, a linear classifier can be learned by maximizing the hard margin, i.e., a hard margin SVM.

When the training data is not linearly separable but can be approximately linearly separable, a linear classifier can be learned by soft margin maximization, i.e., soft margin SVM. Soft margin maximization means allowing the SVM to make errors on a small number of samples, i.e., relaxing the previous hard margin minimization condition a bit.

When the training data is linearly inseparable, a nonlinear SVM can be learned by using the kernel trick and soft interval maximization.

The advantages of SVM are:

1. Since SVM is a convex optimization problem, the solution obtained must be globally optimal and not locally optimal.

2. It is applicable not only to linear linear problems but also to nonlinear problems (using kernel tricks).

3. Data with high-dimensional sample space can also be used with SVM because the complexity of the data set depends only on the support vector and not on the dimensionality of the data set, which in a sense avoids the "dimensionality disaster".

4. The theoretical basis is better (e.g., neural networks are more like a black box).

The disadvantage of SVM is that:

1. The solution of a quadratic programming problem will involve the computation of a matrix of order m (m is the number of samples), so SVMs are not suitable for very large data sets. (SMO algorithm can alleviate this problem)
2. It is only applicable to binary classification problems. (The generalized SVR of SVM is also applicable to regression problems; multiple classification problems can be solved by combining multiple SVMs)

#### MLP

Multilayer perceptron (MLP) is a forward-structured artificial neural network that contains an input layer, an output layer, and multiple hidden layers. 

The basic unit of computation in a neural network is the neuron, generally called a "node" or "unit". Each node can receive inputs from other nodes, or from external sources, and then compute the output. Each input has its own weight (i.e., `w`) and bias (i.e., `b`), with the weight used to regulate the magnitude of the effect of that input on the output and the bias providing a trainable constant value for each node (in addition to the normal input received by the node).

Before the result is output by the neuron, it is processed by an activation function. The activation function generally uses a nonlinear function, whose role is to introduce nonlinearity into the output of the neuron. Since most real-world data are nonlinear, we want the neuron to learn a nonlinear representation of the function. If the activation function remains linear, then the entire function model remains linear according to the principle of linear superposability, and then it cannot solve the problems we encounter, so it is important that the activation function be nonlinear. Some of the common activation functions are as follows.

1. Sigmoid

   Enter a real value $x$ and output a value between 0 and 1.
   $$
   \sigma(x) = \frac{1}{1 + e^{-x}}
   $$

2. tanh

   Input a real value of $x$ and output a value between [-1,1].
   $$
   tanh(x) = \frac{1-e^{-2x}}{1+e^{-2x}} = 2 \sigma(2x) - 1
   $$
   
3. ReLU

   ReLU stands for corrected linear unit. Input a real value $x$, if $x > 0$, output $x$, otherwise output 0.
   $$
   ReLU(x) = max(0, x)
   $$

When a complex feedforward neural network is trained on a small data set, it tends to cause overfitting. To prevent overfitting, the performance of the neural network can be improved by blocking the joint action of feature detectors. Dropout can be used as a trick alternative for training deep neural networks. Overfitting can be significantly reduced by ignoring half of the feature detectors in each training batch (leaving half of the hidden layer nodes with a value of 0). This approach reduces interactions between feature detectors (hidden layer nodes), where detector interactions are those where some detectors depend on other detectors to function. Dropout works by letting the activation value of a certain neuron stop working with a certain probability p during forward propagation, which makes the model more generalizable because it does not depend too much on certain local features.

#### Logistic regression

Logistic regression solves dichotomous problems and is used to express the probability of something happening. The principle of logistic regression is to map the result of linear regression $(-∞,∞)$ to $(0,1)$ using a logistic function. The logic function here is the sigmoid function we mentioned earlier.

The advantages of logistic regression are:

1. Simplicity of implementation and wide application to industrial problems.
2. Very low computational effort for classification, high speed and low storage resources.
3. Convenient observation of sample probability scores.
4. Multicollinearity is not a problem for logistic regression, which can be combined with L2 regularization to solve the problem.
5. Computationally inexpensive and easy to understand and implement.

The disadvantage of logistic regression is that:

1. Logistic regression does not perform very well when the feature space is large.
2. Prone to underfitting and generally not very accurate.
3. Can only handle two classification problems (softmax derived from this can be used for multi-classification) and must be linearly separable.
4. Requires transformation for non-linear features.

#### Decision tree

The decision tree algorithm is a supervised learning algorithm based on if-then-else rules, it uses a tree-like structure and uses layers of inference to achieve the final classification. For prediction, a judgment is made at the node of the tree with a certain attribute value, and based on the judgment result, it decides which branch node to enter until it reaches the leaf node to get the classification result.

The learning process of a decision tree consists of 3 steps:

1. Feature selection

   Feature selection determines which features are used to make judgments. In the training dataset, there may be many attributes for each sample, and different attributes may be more or less useful. Thus, the role of feature selection is to filter out the features that are more relevant to the classification results, i.e., the features with higher classification power.

   The commonly used criterion in feature selection is: information gain.

2. Decision tree generation

   Once the features are selected, the nodes are triggered from the root node, and the information gain of all features is calculated for the nodes.

3. Decision tree pruning

   The main purpose of pruning is to combat "overfitting" by actively removing some branches to reduce the risk of overfitting.

The advantages of decision tree are:

1. Decision tree is easy to understand and interpret, can be analyzed visually, and rules can be easily extracted.
2. Can handle both nominal and numerical data.
3. Being more suitable for handling samples with missing attributes.
4. The ability to handle uncorrelated features.
5. Relatively fast operation when testing datasets.
6. The ability to produce feasible and effective results for large data sources in a relatively short period of time.

The disadvantage of decision tree is that:

1. Prone to overfitting (random forest can reduce overfitting to a large extent).
2. Tend to ignore the interconnection of attributes in the dataset.
3. For data with inconsistent number of samples in each category, in the decision tree, different decision criteria will bring about different attribute selection preferences when classifying attributes; the information gain criterion has a preference for the more desirable attributes (typically represented by the ID3 algorithm), while the gain rate criterion (CART) has a preference for the less desirable attributes, but CART does not simply use the gain rate for attribute classification directly and diligently, but uses a heuristic rule, (whenever information gain is used, it has this drawback, such as RF).
4. The ID3 algorithm calculates the information gain with a result biased towards features with more values.

#### Random forest

Random Forest is a supervised algorithm, which is an integrated learning method consisting of many decision trees, with no correlation between the different decision trees.

When we perform the classification task, new input samples enter and let each decision tree in the forest judge and classify them separately. Each decision tree will get a classification result of its own, and whichever classification result of the decision tree has the most classifications, then the random forest will take that result as the final result.

There are 4 steps to constructing a random forest :

1. A sample with a sample size of $N$ is randomly selected $N$ times, one at a time, to obtain $N$ samples, which are used to train a decision tree.
2. When each sample has $M$ attributes, $m$ attributes are randomly selected from these $M$ attributes when each node of the decision tree needs to be split, satisfying the condition $m << M$. Then some strategy (e.g. information gain) is used to select 1 attribute from these $m$ attributes as the splitting attribute for that node.
3. Each node in the decision tree is split according to step 2 (if the next attribute selected for the node is the same attribute that was used in the splitting of its parent node, then the node has already reached the leaf node and does not need to be split), until it is no longer possible to split. Note that there is no pruning during the whole decision tree formation process.
4. Follow steps 1 to 3 to build a large number of decision trees, which will form a random forest.

The advantages of random forest are:

1. It can come up with very high dimensional (many features) data without dimensionality reduction and without having to do feature selection.
2. It can determine the importance of features.
3. It can determine the interaction between different features.
4. It is not prone to over-fitting.
5. It is faster to train and easier to make parallel methods.
6. It can balance the error for unbalanced datasets.
7. If a large proportion of features are missing, accuracy can still be maintained.

The disadvantage of random forest is that:

1. Random forests have been shown to overfit on certain noisy classification or regression problems.
2. For data with attributes that have different values, attributes with more divided values will have a greater impact on the random forest, so the attribute weights produced by the random forest on such data are not credible.

#### KNN

The KNN (K-Nearest Neighbor) algorithm is a basic classification and regression algorithm which belongs to the category of classification methods in supervised learning. The general idea is formulated as follows.

1. Given a training set `M` and a test object `n`, where the object is a vector consisting of an attribute value and an unknown category label.
2. Calculate the distance (typically Euclidean distance) or similarity (typically cosine similarity) between object `m` and each object in the training set, and determine the list of nearest neighbors.
3. The category with the largest number of occupants in the nearest neighbor list is awarded to the test object `z`.
4. In general, we select only the top $K$ most similar data in the training sample, which is where "k" comes from in the k-nearest neighbor algorithm.

$K$ is generally determined by cross-validation. As a rule of thumb, generally $K$ is lower than the square root of the number of training samples.

The advantages of KNN are:

1. Simple to use and very easy to figure out the basic principles.
2. The algorithm is inert, and the model training time is fast; KNN algorithm does not have an explicit data training process, or it does not need to train data at all, and can directly judge the test object.
3. It is suitable for multi-classification problems.

The disadvantage of KNN is that:

1. High memory requirements for the computer: because it stores the entire training data, the performance is low.
2. Poor interpretability, cannot give certain rules of interpretation for the results.

#### Naive Bayes

Naive Bayes is a classification algorithm based on Bayes' theorem

According to the Bayesian equation:
$$
P(B|A) = \frac{P(A|B)P(B)}{P(A)}
$$
we can have:
$$
P(\text{calss}|\text{feature}) = \frac{P(\text{feature}|\text{calss})P(\text{calss})}{P(\text{feature})}
$$
This is the principle of Naive Bayes.

The main advantages of the Naive Bayes are:

1. This model originates from classical mathematical theory and has stable classification efficiency.
2. It performs well on small-scale data, can handle multiple classification tasks, and is suitable for incremental training, especially when the amount of data exceeds the memory, we can go to incremental training in batches.
3. Less sensitive to missing data and simpler algorithm, often used for text classification.

The main disadvantages of Naive Bayes are:

1. Theoretically, this model has the smallest error rate compared with other classification methods. However, this is not always the case in practice, because the assumption that attributes are independent of each other for a given output category is often not valid in practice, and the classification is not effective when the number of attributes is large or the correlation between attributes is high. In the case of small correlation between attributes, the performance of Naive Bayes is the best. 
2. The prior probability needs to be known, and the prior probability very often depends on the assumptions, and there can be many kinds of models for the assumptions, so at some times the prediction will be poor due to the assumed prior model.