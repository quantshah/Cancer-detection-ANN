%%Loading data

data=load("traindata.csv");
%=========================================================================================================
%Dividing the training set into random 70/30 ratio. x1 is the training set and x2 is the test set

%=========================================================================================================


trainaccuvec=[]; %%Initializing training and testing accuracy vector to find mean accuracy for n number runs
testaccuvec=[];


for i=1:10 %%Carrying out 5 iterations of testing using random splits of dataset
	[x1 x2]=divideTrainset(data);
	X=x1(:,[1 2 3 4 5 ]);
	y=x1(:,6);
	

	for i=1:size(y)(1)  %%Changing the label '0' to 2
		if(y(i)==0)
			y(i)=2;
		endif
	end

	input_layer_size=5;
	num_hidden_layer=1;
	hidden_layer_size=	250;
	num_labels=2;
	m=size(X,1);

	

	fprintf('\nInitializing Neural Network Parameters ...\n')

	initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
	initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

	% Unroll parameters
	initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];


	fprintf('\nTraining Neural Network... \n')

	%  After you have completed the assignment, change the MaxIter to a larger
	%  value to see how more training helps.
	options = optimset('MaxIter', 600);

	%  You should also try different values of lambda
	lambda = 2;

	% Create "short hand" for the cost function to be minimized
	costFunction = @(p) nnCostFunction(p, ...
	                                   input_layer_size, ...
	                                   hidden_layer_size, ...
	                                   num_labels, X, y, lambda);

	% Now, costFunction is a function that takes in only one argument (the
	% neural network parameters)
	[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

	% Obtain Theta1 and Theta2 back from nn_params
	Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
	                 hidden_layer_size, (input_layer_size + 1))

	Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
	                 num_labels, (hidden_layer_size + 1))



	pred = predict(Theta1, Theta2, X);

	fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);
	trainaccu=mean(double(pred == y)) * 100;
	trainaccuvec=[trainaccuvec;trainaccu];

	fprintf('\nTesting\n' );
	xtest=x2(:,[1 2 3 4 5]);
	ytest=x2(:,[6]);

	for i=1:size(ytest)(1) %%Changing '0' in the last column to '2' 
		if(ytest(i)==0)
			ytest(i)=2;
		endif
	end
	predvec=[];
	for i=1:size(ytest)
		pred=predict(Theta1,Theta2,xtest(i,:));
		ytest(i);
		predvec=[predvec;pred];
	end

	
	fprintf('\nTest Set Accuracy: %f\n', mean(double(predvec == ytest)) * 100);
	testaccu=mean(double(predvec == ytest)) * 100;
	testaccuvec=[testaccuvec;testaccu];

end

fprintf("\nThe mean of training accuracies\n");
mean(trainaccuvec)

fprintf("\nThe mean of testing accuracies\n");
mean(testaccuvec)





	
