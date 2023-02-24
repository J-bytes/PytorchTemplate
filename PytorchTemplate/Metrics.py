import numpy as np

import timm.utils.metrics

from sklearn.metrics import f1_score,recall_score,precision_score,roc_auc_score


from PytorchTemplate import names

class Metrics:
    def __init__(self):
        self.num_classes = len(names)
        self.thresholds = [0.5,]*self.num_classes
        self.names = names

    # def convert(self,pred):
    #
    #     for i in range(self.num_classes) :
    #         pred[:,i] = np.where(
    #             pred[:,i]<=self.thresholds[i],
    #             pred[:,i]/2/self.thresholds[i],               #if true
    #             1 - (1-pred[:,i])/2/(1-self.thresholds[i])    #if false
    #         )
    #     return pred


        self.convert = lambda x : x #if you wish to use default threshold



    def accuracy3(self, true, pred):
        # n, m = true.shape
        # pred2 = self.convert(pred)
        # pred2 = np.where(pred2 > 0.5, 1, 0)
        #
        # accuracy = 0
        # for x, y in zip(true, pred2):
        #     if (x == y).all():
        #         accuracy += 1
        # accuracy /=n
        accuracy = timm.utils.metrics.accuracy(pred,true, topk=(3,))
        return accuracy
    def accuracy(self, true, pred):
        true = np.argmax(true,axis=1)
        pred = np.argmax(pred,axis=1)
        accuracy = np.sum(true==pred)/len(true)

        return accuracy

    def f1(self, true, pred):

        pred2 = self.convert(pred)

        pred2 = np.where(pred2 > 0.5, 1, 0)


        return f1_score(
            true, pred2, zero_division=0,average="macro"
        )  # weighted??

    def precision(self, true, pred):
        pred = self.convert(pred)
        pred = np.where(pred > 0.5, 1, 0)
        results = precision_score(true, pred, average=None, zero_division=0)

        results_dict = {}
        for item, name in zip(results, self.names):
            results_dict[name] = item
        return results_dict

    def recall(self, true, pred):

        pred = self.convert(pred)
        pred = np.where(pred > 0.5, 1, 0)
        results=recall_score(true, pred, average=None, zero_division=0)
        results_dict={}
        for item,name in zip(results,self.names) :
            results_dict[name] = item
        return results_dict

    def computeAUROC_weighted(self, true, pred):
        fpr = dict()
        tpr = dict()
        outAUROC = dict()
        classCount = pred.shape[1]
        for i in range(classCount):

            # fpr[i], tpr[i], thresholds = roc_curve(true[:, i], pred[:, i],pos_label=1)
            #
            # threshold = thresholds[np.argmax(tpr[i] - fpr[i])]
            # logging.info(f"threshold {self.names[i]} : ",threshold)
            # self.thresholds[i] =threshold
            # try :
            #     auroc =  auc(fpr[i], tpr[i])
            # except :
            #     auroc=0
            try :
                auroc = roc_auc_score(true[:, i], pred[:, i],average="weighted")
            except ValueError:
                auroc = 0
            outAUROC[self.names[i]] = auroc
            if np.isnan(outAUROC[self.names[i]]):
                outAUROC[self.names[i]] = 0

        outAUROC["mean"] = np.mean(list(outAUROC.values()))


        return outAUROC

    def computeAUROC(self, true, pred):
        fpr = dict()
        tpr = dict()
        outAUROC = dict()
        classCount = pred.shape[1]
        for i in range(classCount):

            # fpr[i], tpr[i], thresholds = roc_curve(true[:, i], pred[:, i],pos_label=1)
            #
            # threshold = thresholds[np.argmax(tpr[i] - fpr[i])]
            # logging.info(f"threshold {self.names[i]} : ",threshold)
            # self.thresholds[i] =threshold
            # try :
            #     auroc =  auc(fpr[i], tpr[i])
            # except :
            #     auroc=0
            try :
                auroc = roc_auc_score(true[:, i], pred[:, i],average="weighted")
            except ValueError:
                auroc = 0
            outAUROC[self.names[i]] = auroc
            if np.isnan(outAUROC[self.names[i]]):
                outAUROC[self.names[i]] = 0

        outAUROC["mean"] = np.mean(list(outAUROC.values()))


        return outAUROC

    def metrics(self):
        dict = {
            "accuracy": self.accuracy,

        }
        return dict


metrics = Metrics().metrics()

if __name__=="__main__" :
    from PytorchTemplate import names
    num_classes=len(names)
    metric = Metrics()
    metrics = metric.metrics()
    print(metrics)
    label=np.random.randint(0,2,(10,num_classes))
    pred =np.random.random(size=(10,num_classes))
    for key,metric in metrics.items() :
        metric(label,pred)