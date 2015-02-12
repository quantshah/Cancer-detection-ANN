function [Xnorm] =normalise(X)
Xnorm=[];
for i =1:size(X)(2)
	m=mean(X(:,i));
	diff=max(X(:,i))-min(X(:,i));
	Xnorm(:,i)=(X(:,i).-m)/diff;
end