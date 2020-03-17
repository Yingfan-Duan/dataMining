#### Introduction
This is a python class of best subset regression based on the selection criterion Cp, AIC and K-fold cross validation.

#### Input
- x: explanatory variables(required)
- y: reponse variable(required)
- intercept: fit a model with intercept or not(default True)
- isCp: choose Cp criterion as the standard or not(default True)
- isAIC: choose AIC criterion as the standard or not(default True)
- isCV: choose k-fold CV criterion as the standard or not(default True)

#### Attributes
There are four functions in the class.
- reg(): do the regression and return the parameters, must run first.
- Cp_AIC(): calculate the Cp or AIC.
- KfoldCV(): calculate CV.
- output(): output a dictionary containing the name of selected variables and corresponding parameters based on different criterions.

#### Usage
You need to run the ``reg()`` method first to get the regression parameters. Then ``output()`` can output the whole results.

#### Improving
The default number of folds in KfoldCV is 10, maybe could make it as a controllable parameter.
