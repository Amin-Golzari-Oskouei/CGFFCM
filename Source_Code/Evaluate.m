function EVAL = Evaluate(ACTUAL,PREDICTED)
% This fucntion evaluates the performance of a classification model by
% calculating the common performance measures: Accuracy, Sensitivity,
% Specificity, Precision, Recall, F-Measure, G-mean.
% Input: ACTUAL = Column matrix with actual class labels of the training
%                 examples
%        PREDICTED = Column matrix with predicted class labels by the
%                    classification model
% Output: EVAL = Row matrix with all the performance measures
PREDICTED = (calculate_true_labels(PREDICTED',ACTUAL))';

[c_matrix,Result,RefereceResult]= confusion.getMatrix(ACTUAL',PREDICTED');

% NMI score
nmi = fNMI(PREDICTED',ACTUAL');

EVAL = [Result.Accuracy, Result.F1_score, nmi];
