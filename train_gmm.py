import data_loader
from sklearn.mixture import GaussianMixture
import numpy as np
import util
import argparse

parser = argparse.ArgumentParser(description="Train Gaussian Mixture Model to Classify Speaker")
parser.add_argument('--data_dir', type=str, default="./data", help="Train Dataset Directory")
parser.add_argument('--cov_type', type=str, default="spherical",help="Choose Covariance Type From ['spherical', 'diag', 'tied', 'full']")
parser.add_argument('--max_iter', type=int, default=100, help="Max Number of Iteration")
opt = parser.parse_args()
#print(opt.data_dir)   结果： ./data
dataset = data_loader.load_data(opt.data_dir)
#测试代码：print("dataset的值为")
#测试代码：print(dataset)
X_train, Y_train, X_test, Y_test = data_loader.train_and_test(dataset)
n_classes = 4
# n_classes = np.unique(Y_train)..?
#np.set_printoptions(threshold=np.inf)#用来显示完整的数组
#print(X_train)
estimator = GaussianMixture(n_components=n_classes, covariance_type=opt.cov_type, max_iter=opt.max_iter, random_state=0,
                            verbose=1)
#max_iter==opt.max_iter=1000

estimator.means_init = np.array([X_train[Y_train == i].mean(axis=0) for i in range(n_classes)])

estimator.fit(X_train)
y_train_pred = estimator.predict(X_train)
train_acc = np.mean(y_train_pred.ravel() == Y_train.ravel()) * 100

y_test_pred = estimator.predict(X_test)
test_acc = np.mean(y_test_pred.ravel() == Y_test.ravel()) * 100

print("train accuracy: ", train_acc)
print("test accuracy: ", test_acc)

util.save_model(estimator)





'''
X_train=np.array(x_train)


x1=X_train[y_train==1]
# --------------------------GMM----------------------------------------------------
# 在这里准备开始搞GMM了，这里用了2个GMM模型

estimator = GaussianMixture(n_components=n_classes,
                   covariance_type='full', max_iter=200, random_state=0,tol=1e-5)

estimator.means_init = np.array([X_train[y_train == i].mean(axis=0) for i in range(n_classes)])    #在这里初始化的，这个值就是我们之前kmeans得到的
estimator.fit(X_train)


y_train_pred = estimator.predict(X_train)
train_acc = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
print('开发集train准确率',train_acc)

print('-'*60)

y_test_pred = estimator.predict(X_test)
test_acc = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
print('验证集dev准确率',test_acc)

util.save_model(estimator)


#x1test_p=np.exp(estimator.score_samples(X_test))
# print(x1test_p[:5],x1test_p.shape)
# print(x2test_p[:5],x2test_p.shape)
# print(y_test[:5],y_test.shape)



'''

'''
estimators = {cov_type: GaussianMixture(n_components=n_classes, covariance_type=cov_type,
        max_iter=100, random_state=0) for cov_type in ['spherical', 'diag', 'tied', 'full']}

for index, (name, estimator) in enumerate(estimators.items()):
    estimator.means_init = np.array([X_train[Y_train==i].mean(axis=0) for i in range(n_classes)])
    print(name, "cov")

    estimator.fit(X_train)

    y_train_pred = estimator.predict(X_train)
    train_acc = np.mean(y_train_pred.ravel() == Y_train.ravel()) * 100

    y_test_pred = estimator.predict(X_test)
    test_acc = np.mean(y_test_pred.ravel() == Y_test.ravel()) * 100

    print("train accuracy :", train_acc)
    print("test accuracy :", test_acc)
'''

'''
class sklearn.mixture.GaussianMixture(n_components=1, covariance_type='full', tol=0.001, reg_covar=1e-06, max_iter=100, 
n_init=1, init_params='kmeans', weights_init=None, means_init=None, precisions_init=None, random_state=None, warm_start=False,
 verbose=0, verbose_interval=10)
 
1. n_components: 混合高斯模型个数，默认为 1 
2. covariance_type: 协方差类型，包括 {‘full’,‘tied’, ‘diag’, ‘spherical’} 四种，
full 指每个分量有各自不同的标准协方差矩阵，完全协方差矩阵（元素都不为零）， tied 指所有分量有相同的标准协方差矩阵（HMM 会用到），
diag 指每个分量有各自不同对角协方差矩阵（非对角为零，对角不为零）， 
spherical 指每个分量有各自不同的简单协方差矩阵，球面协方差矩阵（非对角为零，对角完全相同，球面特性），默认‘full’ 完全协方差矩阵 
3. tol：EM 迭代停止阈值，默认为 1e-3. 
4. reg_covar: 协方差对角非负正则化，保证协方差矩阵均为正，默认为 0 
5. max_iter: 最大迭代次数，默认 100 
6. n_init: 初始化次数，用于产生最佳初始参数，默认为 1 
7. init_params: {‘kmeans’, ‘random’}, defaults to ‘kmeans’. 初始化参数实现方式，默认用 kmeans 实现，也可以选择随机产生 
8. weights_init: 各组成模型的先验权重，可以自己设，默认按照 7 产生 
9. means_init: 初始化均值，同 8 
10. precisions_init: 初始化精确度（模型个数，特征个数），默认按照 7 实现 
11. random_state : 随机数发生器 
12. warm_start : 若为 True，则 fit（）调用会以上一次 fit（）的结果作为初始化参数，适合相同问题多次 fit 的情况，能加速收敛，默认为 False。 
13. verbose : 使能迭代信息显示，默认为 0，可以为 1 或者大于 1（显示的信息不同） 
14. verbose_interval : 与 13 挂钩，若使能迭代信息显示，设置多少次迭代后显示信息，默认 10 次。
'''
