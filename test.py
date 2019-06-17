import  tensorflow as tf
import numpy as np
arr=tf.constant([[[1,2,3],[4,5,6]],[[3,-2,-3],[-4,-5,-6]]])

con = tf.constant([True,False,True,False])

newarr=tf.where(condition=arr>0)
with tf.Session() as sess:
    print(arr.shape)
    ll = sess.run(newarr)
    print(ll)
    l2 = ll[np.where(ll[:,1] == 0)]
    print(ll[1:2])
    xu = np.unique(ll[:,0])
    # print(xu)#获取索引
    # num = tf.gather(arr,xu)
    # print(sess.run(num))
