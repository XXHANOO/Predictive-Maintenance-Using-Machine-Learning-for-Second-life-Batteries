% Load and preprocess the data
data = readtable('battery_data.csv'); % Replace with actual file
% Assume columns: voltage, current, temperature, SoH
data_scaled = normalize(table2array(data)); % Normalize data

sequenceLength = 30; % Length of LSTM sequences
X = []; % Input features
y = []; % Target SoH values

% Create sequences
for i = 1:(size(data_scaled, 1) - sequenceLength)
    X = cat(3, X, data_scaled(i:i+sequenceLength-1, 1:end-1));
    y = [y; data_scaled(i+sequenceLength, end)];
end

% Split data into training and testing sets
numTrain = floor(0.8 * size(X, 3));
XTrain = X(:, :, 1:numTrain);
yTrain = y(1:numTrain);
XTest = X(:, :, numTrain+1:end);
yTest = y(numTrain+1:end);

% Define LSTM network
layers = [
    sequenceInputLayer(size(XTrain, 1))
    lstmLayer(64, 'OutputMode', 'sequence')
    dropoutLayer(0.2)
    lstmLayer(64)
    dropoutLayer(0.2)
    fullyConnectedLayer(1)
    regressionLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 32, ...
    'ValidationData', {XTest, yTest}, ...
    'Plots', 'training-progress', ...
    'Verbose', 0);

% Train LSTM network
net = trainNetwork(XTrain, yTrain, layers, options);

% Predict and evaluate
yPred = predict(net, XTest);
mseError = mean((yTest - yPred).^2);
disp(['Test MSE: ', num2str(mseError)]);

% Plot actual vs predicted SoH
figure;
scatter(yTest, yPred);
xlabel('Actual SoH');
ylabel('Predicted SoH');
title('Actual vs Predicted SoH');
grid on;
