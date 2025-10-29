from numpy import array, dot, random
from pandas import read_csv
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from time import time
from matplotlib.pyplot import legend, plot, savefig, scatter, show, title, xlabel, ylabel
from warnings import filterwarnings
filterwarnings("ignore")

df, MSERates, splitFracts, Degs, fourthRates, Regressors, trainErrors, testErrors, modelErrors, regularizedErrors, model_trErrors, model_ttErrors, regularized_trErrors, regularized_ttErrors, splitIndexer, Block = read_csv("Q2data.csv"), [(4500, 0.00185), (3750, 0.0000015), (2400, 0.0000000029)], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [1, 2, 3], [0.0000000000028, 0.0000000000028, 0.0000000000028, 0.0000000000028, 0.00000000000285, 0.0000000000028, 0.0000000000027, 0.0000000000025, 0.0000000000025], [], [], [], {}, {}, [], [], [], [], [], False
X, y = array([x for x in df["X"]]), array([y for y in df["Y"]])# Scaler.fit_transform(array([x for x in df["X"]]).reshape(-1, 1))[:,0], Scaler.fit_transform(array([y for y in df["Y"]]).reshape(-1, 1))[:,0]

# Mean Squared Error as Cost Function.
def MSE(X, W, y, n):
    return sum([(y[Index] - dot(W, X[Index])) ** 2 for Index in range(n)]) / n       

# Polynomial Regressions
# Plotting raw
Title = "Data Points"
scatter(X, y, c = "black")
title(Title)
xlabel("$x$")
ylabel("$y$")
savefig(Title, dpi = 1200)
show(block = Block)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42, shuffle = True)
n, m = X_train.shape[0], X_test.shape[0]
for Degree, MSERate in enumerate(MSERates):
    Degree += 1
    # Train features
    Poly = PolynomialFeatures(degree = Degree)
    polyX_train, polyX_test, W, MSE_boundary, learningRate = Poly.fit_transform(X_train.reshape(-1, 1)), Poly.fit_transform(X_test.reshape(-1, 1)), random.rand(1, Degree + 1)[0], MSERate[0], MSERate[1]
    print(f"\n\nPrimary initialized W vector is {W} while degree equals {Degree}.\n\nLearning...")
    Start = time()
    # Checking goodness of value of MSE error function.
    while MSE(polyX_train, W, y_train, n) > MSE_boundary:
        W += learningRate * sum([(y_train[Index] - dot(W, polyX_train[Index])) * polyX_train[Index] for Index in range(n)]) / n
    End = time()
    MSE_opt, MSE_v = MSE(polyX_train, W, y_train, n), MSE(polyX_test, W, y_test, m)
    Regressors.append(dot(Poly.fit_transform(X.reshape(-1, 1)), W))
    trainErrors.append(MSE_opt)
    testErrors.append(MSE_v)
    print(f"\nLearning for fitting {Degree}-degree polynomial curve on 60% of shuffled dataset occured in {End - Start} UTC time unit.\nOptimal value of MSE cost is {MSE_opt} with optimal W vector {W}")

# Plotting learned regressors
for Degree, Regressor in enumerate(Regressors):
    Title =  f"{Degree + 1}-degree Polynomial Fitting."
    scatter(X, Regressor, c = "red")
    scatter(X, y, c = "black")
    title(Title)
    xlabel("$x$")
    ylabel("$y$")
    savefig(Title, dpi = 1200)
    show(block = Block)

Title = "Error Curves"
scatter(Degs, trainErrors, color = "blue")
scatter(Degs, testErrors, color = "red")
plot(Degs, trainErrors, linestyle = "-", color = "blue", label = "Train Error")
plot(Degs, testErrors, linestyle = "-", color = "red", label = "Test Error")
# title
title(Title)
# x label
xlabel("Degree of Polynomial")
# y label
ylabel("Errors")
legend(loc = "best")
savefig(Title, dpi = 1200)
show(block = Block)

# Fourth degree
# Scratch form
Poly, MSE_boundary, degreePlus = PolynomialFeatures(degree = 4), 1300, 5
polyX = Poly.fit_transform(X.reshape(-1, 1))
for splitFract, learningRate in enumerate(fourthRates):
    splitFract, Title = (splitFract + 1) / 10, f"4-degree Polynomial Fitting for {int(splitFract * 10)}th dataset split."
    Percent = int(splitFract * 100)
    print(f"\n\n4-degree polynomial fitting started!\n{Percent}% of whole dataset used as test set.")
    X_train, X_test, y_train, y_test = train_test_split(polyX, y, test_size = splitFract, random_state = 42, shuffle = True)
    W, n, m = random.rand(1, degreePlus)[0], X_train.shape[0], X_test.shape[0]
    print(f"Primary initialized W vector is {W}.\n\nLearning...")
    Start = time()
    # Goodness of value of MSE error function.
    while MSE(X_train, W, y_train, n) > MSE_boundary:
        W += learningRate * sum([(y_train[Index] - dot(W, X_train[Index])) * X_train[Index] for Index in range(n)]) / n
        #print(W)
    End = time()
    model_train_error = MSE(X_train, W, y_train, n)
    print(f"\nLearning for fitting 4-degree polynomial curve on {100 - Percent}% of dataset occured in {End - Start} UTC time unit.\nOptimal value of MSE cost is {model_train_error} with optimal W vector {W}")

    # Plotting 4-degree polynomial curve
    scatter(X, dot(polyX, W), c = "red")
    scatter(X, y, c = "black")
    title(Title)
    xlabel("$x$")
    ylabel("$y$")
    savefig(Title, dpi = 1200)
    show(block = Block)

    # Regularization strength, when needed avoiding or decreasing overfitting event.
    regularizedModel = Ridge(alpha = 1e+5).fit(X_train, y_train)
    # Compute Mean Squared Errors and save them for comparison and self-made evaluation phase.
    model_test_error, regularized_train_error, regularized_test_error = MSE(X_test, W, y_test, m), mean_squared_error(y_train, regularizedModel.predict(X_train)), mean_squared_error(y_test, regularizedModel.predict(X_test))
    model_trErrors.append(model_train_error)
    model_ttErrors.append(model_test_error)
    regularized_trErrors.append(regularized_train_error)
    regularized_ttErrors.append(regularized_test_error)
    model_errs, regularized_errs = (model_train_error, model_test_error), (regularized_train_error, regularized_test_error)
    splitIndexer.append([model_errs, regularized_errs])
    modelErrors[abs(model_train_error - model_test_error)], regularizedErrors[abs(regularized_train_error - regularized_test_error)] = model_errs, regularized_errs
    print(f"Mean Squared Error for train set in non-regularized model and regularized model are {model_train_error} and {regularized_train_error}, respectively\nwhile Mean Squared Error for test set in non-regularized model and regularized model are {model_test_error} and {regularized_test_error}, respectively.")

model_best_errScore, regularized_best_errScore, Title = min(list(modelErrors.keys())), min(list(regularizedErrors.keys())), "4-degree Polynomial Fitting Error Curves"
if model_best_errScore < regularized_best_errScore:
    bestErrors = modelErrors[model_best_errScore]
else:
    bestErrors = regularizedErrors[regularized_best_errScore]

for Index, Errors in enumerate(splitIndexer):
    if bestErrors in Errors:
        bestIndex = int(Index + 1)
        break
print(f"\n\nBest overall score (lowest distance of train error and test error) achieved from {bestIndex}th split in which the value of MSE error for train set equals {bestErrors[0]} and the value of MSE error for test set equals {bestErrors[1]}, respectively.")
scatter(splitFracts, model_trErrors, color = "blue")
scatter(splitFracts, model_ttErrors, color = "red")
scatter(splitFracts, regularized_trErrors, color = "yellow")
scatter(splitFracts, regularized_ttErrors, color = "green")

plot(splitFracts, model_trErrors, linestyle = "-", color = "blue", label = "Train Error")
plot(splitFracts, model_ttErrors, linestyle = "-", color = "red", label = "Test Error")
plot(splitFracts, regularized_trErrors, linestyle = "-", color = "yellow", label = "Regularized Based Train Error")
plot(splitFracts, regularized_ttErrors, linestyle = "-", color = "green", label = "Regularized Based Test Error")
# title
title(Title)
# x label
xlabel("Split Fraction")
# y label
ylabel("Errors")
legend(loc = "best")
savefig(Title, dpi = 1200)
show()

print("THE-END!")
