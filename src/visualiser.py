from dependencies import *
from assessment import plot_confusion_matrix

class visualiser:
    def __init__(self, X= None, y= None):
        '''
            Initialize object with the predictors/ response variable
        '''
        self.X= X
        self.y= y
            
    def class_imbalance( self ):
        '''
            Plot the class imbalance as a barplot of the classes with 
            an associated counter.
        '''
        plt.figure(figsize=(8, 8))
        cplot= sns.countplot(x= self.y)
        for p in cplot.patches:
            cplot.annotate( format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
        plt.title('Classes Imbalance')
        plt.show()
        
    def corr_heatmap( self ): 
        '''
            Plot a heatmap with the correlations of the predictors in X.
        '''
        fig, ax = plt.subplots(figsize=(20,10))         
        # calculate correlations
        corr = self.X.corr()
        # mask to show only the lower triangle
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        
        with sns.axes_style("white"):
            sns.heatmap(data=corr, mask=mask, cmap='PRGn', annot_kws={'size':30}, cbar_kws={'label': 'colorbar title'}, vmin= -1, vmax=1, linewidths=.5, ax=ax)
        ax.set_title("Predictors Heatmap", fontsize=14)
        plt.show()
        
    # From the assessment module; plots the confusion matrix
    def confusion_matrix(self, y_pred, classes, normalize=False): plot_confusion_matrix(y_true= self.y, y_pred= y_pred, classes= classes, normalize= False)
        
    def f1( self, y_pred):
        '''
            Creates instance variable with the harmonic mean and displays it. 
        '''
        self.f1= f1_score(self.y, y_pred)
        print(f"F1 score is: {self.f1}")
        
    def areaUnderPR(self, y_pred ):
        '''
            plots the Precision-Recall curve and displays the area under the curve (average precision).
        '''
        average_precision = average_precision_score(self.y, y_pred)
        precision, recall, _ = precision_recall_curve(self.y, y_pred)

        step_kwargs = ({'step': 'post'}
                       if 'step' in signature(plt.fill_between).parameters
                       else {})
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format( average_precision) )
        
    