import  tensorflow as tf
import numpy as np
arr=tf.constant([[[1,2,3],[4,5,6]],[[3,-2,-3],[-4,-5,-6]]])
arr2=tf.constant([[1,2,3],[4,5,6],[3,-2,-3],[-4,-5,-6],[1,2,3],[4,5,6],[3,-2,-3],[-4,-5,-6]])
index = []
with tf.device('/gpu:0'):
    newarr=tf.where(arr>0)
    arr1 = newarr[:,2]
    # xu = tf.where(condition=tf.equal(arr1,2),x=arr1,y=-arr1-1)
    xu = tf.where(condition=tf.equal(arr1,2))
    index.append(xu)
    xu2 = tf.gather(arr1,index)
    # num = tf.gather(arr2,index)
with tf.Session() as sess:
    print(arr2)
    print(sess.run(newarr))
    print(sess.run(xu2))
    xuu = sess.run(xu)
    print(xuu)
    # num = sess.run(num)
    # print(num)#获取索引
    # num = tf.gather(arr,xu)
    # print(sess.run(num))
