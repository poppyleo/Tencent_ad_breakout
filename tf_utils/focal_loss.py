import tensorflow as tf
from tensorflow.python.ops import array_ops


def focal_loss(logits, labels, batch_size, alpha=[[1], [1]], epsilon=1.e-7,
               gamma=2.0,
               multi_dim=False):
    '''
    :param logits:  [batch_size, n_class]
    :param labels: [batch_size]  not one-hot !!!
    :return: -alpha*(1-y)^r * log(y)
    它是在哪实现 1- y 的？ 通过gather选择的就是1-p,而不是通过计算实现的；
    logits softmax之后是多个类别的概率，也就是二分类时候的1-P和P；多分类的时候不是1-p了；

    怎么把alpha的权重加上去？
    通过gather把alpha选择后变成batch长度，同时达到了选择和维度变换的目的

    是否需要对logits转换后的概率值进行限制？
    需要的，避免极端情况的影响
    # focal loss with multi label
    # 注意，alpha是一个和你的分类类别数量相等的向量；
    针对输入是 (N，P，C )和  (N，P)怎么处理？
    先把他转换为和常规的一样形状，（N*P，C） 和 （N*P,）

    bug:
    ValueError: Cannot convert an unknown Dimension to a Tensor: ?
    因为输入的尺寸有时是未知的，导致了该bug,如果batchsize是确定的，可以直接修改为batchsize

    '''

    if multi_dim:
        logits = tf.reshape(logits, [-1, logits.shape[2]])
        labels = tf.reshape(labels, [-1])

    # (Class ,1)
    alpha = tf.constant(alpha, dtype=tf.float32)

    labels = tf.cast(labels, dtype=tf.int32)
    logits = tf.cast(logits, tf.float32)
    # (N,Class) > N*Class
    softmax = tf.reshape(tf.nn.softmax(logits), [-1])  # [batch_size * n_class]
    # (N,) > (N,) ,但是数值变换了，变成了每个label在N*Class中的位置
    labels_shift = tf.range(0, batch_size) * logits.shape[1] + labels
    # labels_shift = tf.range(0, batch_size*32) * logits.shape[1] + labels
    # (N*Class,) > (N,)
    prob = tf.gather(softmax, labels_shift)
    # 预防预测概率值为0的情况  ; (N,)
    prob = tf.clip_by_value(prob, epsilon, 1. - epsilon)
    # (Class ,1) > (N,)
    alpha_choice = tf.gather(alpha, labels)
    # (N,) > (N,)
    weight = tf.pow(tf.subtract(1., prob), gamma)
    weight = tf.multiply(alpha_choice, weight)
    # (N,) > 1
    loss = -tf.reduce_mean(tf.multiply(weight, tf.log(prob)))
    return loss


# def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
#     r"""Compute focal loss for predictions.
#         Multi-labels Focal loss formula:
#             FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
#                  ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
#     Args:
#      prediction_tensor: A float tensor of shape [batch_size, num_anchors,
#         num_classes] representing the predicted logits for each class
#      target_tensor: A float tensor of shape [batch_size, num_anchors,
#         num_classes] representing one-hot encoded classification targets
#      weights: A float tensor of shape [batch_size, num_anchors]
#      alpha: A scalar tensor for focal loss alpha hyper-parameter
#      gamma: A scalar tensor for focal loss gamma hyper-parameter
#     Returns:
#         loss: A (scalar) tensor representing the value of the loss function
#     """
#     sigmoid_p = tf.nn.sigmoid(prediction_tensor)
#     zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
#
#     # For poitive prediction, only need consider front part loss, back part is 0;
#     # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
#     pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
#
#     # For negative prediction, only need consider back part loss, front part is 0;
#     # target_tensor > zeros <=> z=1, so negative coefficient = 0.
#     neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
#     per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
#                           - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
#     return tf.reduce_sum(per_entry_cross_ent)

#

####opt

class_num = 10

weights = [1.0] * class_num


def search_weight(valid_y, raw_prob, init_weight=[1.0] * class_num, step=0.001):
    weight = init_weight.copy()
    from sklearn.metrics import  accuracy_score
    import numpy as np
    f_best = accuracy_score(y_true=valid_y, y_pred=raw_prob.argmax(axis=1))
    flag_score = 0
    round_num = 1
    while (flag_score != f_best):
        print("round: ", round_num)
        round_num += 1
        flag_score = f_best
        for c in range(class_num):
            for n_w in range(0, 2000, 10):
                num = n_w * step
                new_weight = weight.copy()
                new_weight[c] = num
                prob_df = raw_prob.copy()
                prob_df = prob_df * np.array(new_weight)
                f = accuracy_score(y_true=valid_y, y_pred=prob_df.argmax(
                    axis=1))
                if f > f_best:
                    weight = new_weight.copy()
                    f_best = f
                    print(f)
    return weight
