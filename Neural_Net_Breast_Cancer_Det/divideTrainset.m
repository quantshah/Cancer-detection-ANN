%%Divides the training set randomly into two matrices in a 70-30 ratio for testing
function [trainX testX]=divideTrainset(X);

%%Sizes
m=size(X)(1);
n=size(X)(2);

trainRows=int32(0.7*m);
testRows=m-trainRows;

%%Initialization of trainX and testX
trainX=zeros(trainRows,n);
testX=zeros(testRows,n);

randindices=randperm(m,trainRows);
for i =1:trainRows
	trainX(i,:)=X(randindices(i),:);
end
testindices=zeros(testRows,1);
l=1;
for i=1:m
	check=(randindices==i);
	if (check==0)
		testindices(l)=i;
		l+=1;
	endif
end
for i =1:testRows
	testX(i,:)=X(testindices(i),:);
end