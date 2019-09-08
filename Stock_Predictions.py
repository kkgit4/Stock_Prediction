{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>High</th>\n",
       "      <th>Low</th>\n",
       "      <th>Open</th>\n",
       "      <th>Close</th>\n",
       "      <th>Volume</th>\n",
       "      <th>Adj Close</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Date</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>2018-12-24</th>\n",
       "      <td>1003.539978</td>\n",
       "      <td>970.109985</td>\n",
       "      <td>973.900024</td>\n",
       "      <td>976.219971</td>\n",
       "      <td>1590300.0</td>\n",
       "      <td>976.219971</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2018-12-26</th>\n",
       "      <td>1040.000000</td>\n",
       "      <td>983.000000</td>\n",
       "      <td>989.010010</td>\n",
       "      <td>1039.459961</td>\n",
       "      <td>2373300.0</td>\n",
       "      <td>1039.459961</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2018-12-27</th>\n",
       "      <td>1043.890015</td>\n",
       "      <td>997.000000</td>\n",
       "      <td>1017.150024</td>\n",
       "      <td>1043.880005</td>\n",
       "      <td>2109800.0</td>\n",
       "      <td>1043.880005</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2018-12-28</th>\n",
       "      <td>1055.560059</td>\n",
       "      <td>1033.099976</td>\n",
       "      <td>1049.619995</td>\n",
       "      <td>1037.079956</td>\n",
       "      <td>1414800.0</td>\n",
       "      <td>1037.079956</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2018-12-31</th>\n",
       "      <td>1052.699951</td>\n",
       "      <td>1023.590027</td>\n",
       "      <td>1050.959961</td>\n",
       "      <td>1035.609985</td>\n",
       "      <td>1493300.0</td>\n",
       "      <td>1035.609985</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   High          Low         Open        Close     Volume  \\\n",
       "Date                                                                        \n",
       "2018-12-24  1003.539978   970.109985   973.900024   976.219971  1590300.0   \n",
       "2018-12-26  1040.000000   983.000000   989.010010  1039.459961  2373300.0   \n",
       "2018-12-27  1043.890015   997.000000  1017.150024  1043.880005  2109800.0   \n",
       "2018-12-28  1055.560059  1033.099976  1049.619995  1037.079956  1414800.0   \n",
       "2018-12-31  1052.699951  1023.590027  1050.959961  1035.609985  1493300.0   \n",
       "\n",
       "              Adj Close  \n",
       "Date                     \n",
       "2018-12-24   976.219971  \n",
       "2018-12-26  1039.459961  \n",
       "2018-12-27  1043.880005  \n",
       "2018-12-28  1037.079956  \n",
       "2018-12-31  1035.609985  "
      ]
     },
     "execution_count": 85,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import datetime\n",
    "import seaborn as sns\n",
    "import numpy as np\n",
    "import pandas_datareader.data as web\n",
    "from pandas import Series, DataFrame\n",
    "from sklearn import preprocessing\n",
    "from sklearn import model_selection\n",
    "from sklearn import linear_model\n",
    "from sklearn.model_selection import train_test_split\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.linear_model import Ridge\n",
    "from sklearn.linear_model import Lasso\n",
    "\n",
    "start = datetime.datetime(2010, 1, 1)\n",
    "end = datetime.datetime(2018, 12, 31)\n",
    "\n",
    "df = web.DataReader(\"GOOG\", 'yahoo', start, end)\n",
    "df.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "metadata": {},
   "outputs": [],
   "source": [
    "def prepare_data(df,forecast_col,forecast_out,test_size):\n",
    "    label = df[forecast_col].shift(-forecast_out);#creating new column called label with the last 5 rows are nan\n",
    "    X = np.array(df[[forecast_col]]); #creating the feature array\n",
    "    X = preprocessing.scale(X) #processing the feature array\n",
    "    X_lately = X[-forecast_out:] #creating the column i want to use later in the predicting method\n",
    "    X = X[:-forecast_out] # X that will contain the training and testing\n",
    "    label.dropna(inplace=True); #dropping na values\n",
    "    y = np.array(label)  # assigning Y\n",
    "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size) #cross validation \n",
    "\n",
    "    response = [X_train,X_test ,y_train, y_test , X_lately];\n",
    "    return response;\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {},
   "outputs": [],
   "source": [
    "forecast_col = 'Close'#choosing which column to forecast\n",
    "forecast_out = 5 #how far to forecast \n",
    "test_size = 0.2; #the size of my test se"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{'test_score': 0.9944449329037267, 'forecast_set': array([ 977.23057564, 1040.36643038, 1044.77919597, 1037.99034458,\n",
      "       1036.52279444])}\n"
     ]
    }
   ],
   "source": [
    "X_train, X_test, y_train, y_test , X_lately =prepare_data(df,forecast_col,forecast_out,test_size); #calling the method were the cross validation and data preperation is in\n",
    "\n",
    "learner = linear_model.LinearRegression(); #initializing linear regression model\n",
    "\n",
    "learner.fit(X_train,y_train); #training the linear regression model\n",
    "score=learner.score(X_test,y_test);#testing the linear regression model\n",
    "\n",
    "forecast= learner.predict(X_lately); #set that will contain the forecasted data\n",
    "\n",
    "response={};#creting json object\n",
    "response['test_score']=score; \n",
    "response['forecast_set']=forecast;\n",
    "\n",
    "print(response);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "metadata": {},
   "outputs": [],
   "source": [
    "lm = LinearRegression()\n",
    "lm.fit(X_train,y_train)\n",
    "predictions = lm.predict( X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0, 0.5, 'Predicted Y')"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.scatter(y_test,predictions)\n",
    "plt.xlabel('Y Test')\n",
    "plt.ylabel('Predicted Y')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[<matplotlib.lines.Line2D at 0x1ee3f7e12e8>]"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAX0AAAD4CAYAAAAAczaOAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjAsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+17YcXAAAgAElEQVR4nO2dd5wTxdvAv5NL7g6QpnRQKVIEAUVE7AoqYBd7ReUVe/lZsXdFEVBUVFARLIhdFBQRQUWR3gQEjiK993Ilybx/ZJNL2SS7yW7azdcPXrI7O/tkdvbZZ5955hkhpUShUCgUFQNHugVQKBQKRepQSl+hUCgqEErpKxQKRQVCKX2FQqGoQCilr1AoFBUIZ7oFiEWtWrVk48aN0y2GQqFQZBWzZs3aKqWsrbcvo5V+48aNmTlzZrrFUCgUiqxCCPFftH3KvaNQKBQVCKX0FQqFogKhlL5CoVBUIJTSVygUigqEUvoKhUJRgVBKX6FQKCoQSukrFApFBUIpfYVCobAZKSVfzFxDcZkn3aIopa9QKDRW/g4/PxGzyJRlW5mwaFOKBModJi7ezINfzmfAz0vSLUpmz8hVKBQpYuM/MOoqqNsmZrFr358GwKp+56ZCqpxhd3EZAFv2lKRZEmXpKxSKPRvh0yvAWQiXvJ9uaXISIXx/M2GdQqX0M5Qd+0o5UJp+/18ms3rbfo54dBxvT17Od3PXpVuc7KR0P4y6EnavhdMeghqHpluinEQg0i1CAKX0LcTt8fLptNV4vMk/z495bgLnvznFAqlyl+/nr8ftlbz807/c89lcU8eWebwMmVxEibsCP1i9XvimD6yf4/s+8Vnftixhf6mb96esxGvB/ZYqMmFJcqX0LWTE1P949JsFfPx31AR3pijavNeSesywp7iMXfvLbD3HV7PW8sP89baeIx4f//0fr/y0hKG/rQhs8z085qRRqhTz67Ow+Pvy7z1eBkf2qIR+P/7Lcz8s4udFG9MtSlTKPF4a9x3Lh3+tAjLDvaMGci1i0pLNLFi7E4BdB+xVmnbS7pmfkdLegbr7v5gHwHntGth2jnjs11xn+4NC6N6evByA1688Ji0ypZRZH8KUQeXfT7kfjrk2beIkgv8+O5ABYZDR2F/ik23ump1plqQcpfQt4sbhM9ItgiVkwutnLDxeiUOAECIwOKYwycJv4Pt7yr+36QlnPK5f9vf+4KoMJ9yRGtlMkO7L/93cdRx9aA0OP6SK4WNkBtxg2fMul0UYva7/bdvH+p0H7BUmh9i8u5hmj44LuM/C23nl1n1pkCrLWPAlfHFD+fdGx8FFQ/TdOt/fC78+D+MfTZl4iZAuPXrPZ3M5d3CccbewJ1P6Vb5S+rYwavpqGvcdy/Z9pTHLndZ/Mif2+zVFUqUOr1fS5dXJfD/PWr/96u37Afh2rn692/amPwY6o/nrDfiqd/n3GofDlaNg5xpYFaa8Jj4Ls4b7Pv9vUepkNIHIgFe9vSVucwdkgNaPq/SFEB8IITYLIf4J2tZfCPGvEGK+EOIbIUSNoH2PCCGKhBBLhBDdgrZ317YVCSH6Wv9TMoeNu4sBWLejYlrxxW4PK7bu46Ev59tSv/8VOQPu+ezAXQrf3QE/B7lwCqrDNV/Ang3w/pkwuV/5vj8Hwx8DfJ/vnAnVG6ZMVI9XZoQLxC5kBmh9I5b+h0D3sG0TgKOklO2ApcAjAEKI1sCVQBvtmCFCiDwhRB7wFtADaA1cpZXNOKSUvPbLUtZlidtl696SjA1ZO1Dm4ccFGyyrz+++ycxfm6HsXg8fngNzPi7fJvLgio98nz+62Neg5w70fZ85HCZoqRj6TIZazVMoLDR7dBz3fT7P1DGZ/IwIN0wyQda4Sl9K+TuwPWzbz1JK/3vN30Aj7fOFwGdSyhIp5UqgCOik/SuSUq6QUpYCn2llM47lW/by2i/LuOWjzF+Qff3OA3R8/hfenFSU8nNLKXl1/JK4YxJ3jbIuBPJB7c0hFVPZZ/23g8Z9x7Jhl+/37Sku48+irbaf11JW/QnvngZrw4IMLhgMNQ6DkRfCgR1w+YdQu4XP3//Dvb4yvb6HBumJYvpmjrGJdul60Rvx1yqeHrPQUNlwGbNC6RvgJuBH7XNDYE3QvrXatmjbIxBC9BFCzBRCzNyyZYsF4sHiDbtp3Hcsvy2NX5/faC4ps2eSyluTirjbIkW4YZfPjTRpyWZL6jPDwvW7eXNSEXd+OpuizXv43+i5uD3ZM7EnHv7B4qnLtwFw96g5XPPetIzInRIXKeHvd2DkBbAvrG+cfB80PcO3b88GX2x+sy7w79hyf//lH0GTU20Tb+mmPYycusq2+u3mqTELA3H3Zpmxajs74oz12U1SSl8I8RjgBj7xb9IpJmNsj9wo5VApZUcpZcfatWsnIx7gG1SctsJ34/5iIjtgsdtjaRqEMo+XnftL6T9+CWMsGODcvLuYr2evtUCyxPBbLCVuL3ePmss3c9bx78Y9QGZNOU8W/+9cusk3US7jZ/CW7oevb4afHobKh4Tua3MxHH+rT+HvXA3H/R90uhmKfoHPrvaVOf91aH2BrSKePeh3nvzOmKVshAwwnqMSPti8bV8p17w3LU3S+Eg4Tl8I0Qs4D+gqy0de1gLByTsaAX4NF227rZz+6uRA1IcZ1mw/wJFP/mTZJKV7R89l7Hzr/Nu9hs9g8YbdQPrjlXORVLXpxMWb+GH+BgZdcXTylW1fCaOvhU0LocP18O+48n2NjoNuL8HHPWFbETQ5Dbr386VT/vgSX5kuT8CxNyQvR6rI0o6/SLtv00VClr4QojvwMHCBlDJYo44BrhRCFAghmgDNgenADKC5EKKJECIf32DvmOREN0awwk/nyLmVCh9gy55iS+sL545PZ3PhW3/GLVdRImiCI0qklJYNnvceMdOwDzsmyybA0NNg11q49ANYMwP2a2MQNQ6DnkN91vymf+CQI+DyET5f/4jzfWWOvxVOfSBQ3Yote1m9zbyxlA5yOdrHDoyEbI4CpgIthRBrhRC9gTeBqsAEIcRcIcQ7AFLKhcDnwCLgJ+AOKaVHG/S9ExgPLAY+18rayubd9irGeNj5kEk0RllKSdHmPXHLjZ2/gXkZNHU8k7jj09k0fXRc/IKpwOuF316BTy6D6ofB/02EOR/BlsW+/c5KcPlIeLMTrJ/t29ZzGGxZCsN7+L63udhn9QfRZcBvnNp/kq2it3/m55DvH5nMWZWsG3HMvPX8scyaccNoZKJNZCR65yopZX0ppUtK2UhK+b6U8ggp5aFSyqO1f7cGlX9BStlMStlSSvlj0PZxUsoW2r4X7PpBfn5buoVOL04M2ZZJvubg8YLRM3yTucyMIST6Sz6dvpozB/4eGKBMhLlrdlaYDKD+x3bwQ3bcggxJ8FW8y2e9T3oB2l0OvX+Gv4fA8qAJf1d8BJ9cDt6gfFBT34QPzvZ9bnwKXPJBWl7Zdh0oY8DPS9i53zew+cS3/8Q5wlruHjWH696fbus5Ji+x96GSCDk7I9ef/CwYKyzv7+au46/lxkL3Yj1kXvtlaeDzG7/6Qi63mphRGu0e/Xr2Whr3HRsSIeDxSmb954u6XbB2FwCrtiWesmBM0IxYKdPn4smkh3jK8Xp91n3RBOjxClz8LsweATODFkE57zVfGoXwCJ5/vvL9rdMGrv06rZk13/i1iGe+T27Gb/Bd/fmMNaYNKDu549PZ6RYhgpxV+nZxz2dzuXpY8qPvu4sjp2+Xebys3LrPkI8ymsIbMdX3ihys1N/4dRmXvD2VwROXBbblghs0lWM0meI3llIy4OclbP3tHVgzzRdyuXEBTHoRfnqkvOBJ98KSH2FDjHUGev8Mznz7hY5DohFRAWMj6NIM/tXXx80YUKkm3eNgOav0rcrLUVzmSTj+3KxSemHsYs54dXIg/j4WDhM/b+kmnw9/4ISl/KW5daYUJf7amQlTyW3HRPt+PmMN7/y2PKnTSSn5Y9kWGvcdG3M2+LqdBxj96wwKf3vO55pZ8ZvPh//7KwS0X+sLYe8mWDY++gkvHQ4FB+nuKnUnP99i3IINagH1KKTbfshJpe/1yrjJzozS6omfuPzdqZbUFQ+/Qg6W3euVNO47NsQdBKEPNTMPuE3a4LYdfmm9zmynVZNK906sNn7oq/n0+/HfpOqXEj6b4Zu/OPu/HVHLlXkkT7lG4KIMqtSCxWFBcA06QKWDYd6o6Cc79UE4qmfU3Ze985cp2fW4/ZPZ3Dwy82e1V0RyUunv2F/K+1NWWlbf7NWJRbEYVUqx8nN4tC+v/bIspEw8ZbppdwllOm8oqbYyrDif1yspzuCFMsJZv/MASzbGj5AKRgb+F5uCFT9zbt50Novavrz44VSpVZ4dU48jzoIuUXLna8zTxn0ykU27iwP9OltHdJR7J4Wk67VqwM9LTJUPdp9sjOLqccTpObd+PIvHv4mMhvCmoBGsdv88P3YxrZ74ydI6jeL35Zvx6Z/Y71e6vfZ7zDKv/PQvP/1TPnfDyHX56LeFMM4XS3+ojDK3cdnP+tsBKteCq0fHPU+mUlzm4fgXJ/JwWPbWCuFutBC1cpaN+DujPzrHKMHzfnp9oB9SZsSnP2HxJl4O2+a2ISOn3ZbLZzNW23uCNDBkcugYgBGlf2DCCzRwJpH0rWQPrJsNhx6XeB1ppFSz8P1jBamymAdNWMqO/aU8e+FRqTmhzeSkpR/N/5ru16pohN/vwQogWoIvvd/43dx1IROqUv1z9SyuTG1zs9i9YEdcnb9hHn2cY5M7iaeErb++wW0vDaFk4Q/J1WUzc1ZHH9dItV3/+sRljJxqbuKYETxeyaQlm1MeGZaTSj8aqXbvJDrQaCxkM5Jhf6yIUl9CYhhCL69R8PmsOHc63HLh187KG1MvvUHM6r0e3yIoFvD+tja8VvwE+6eNMFT+3MF/cPEQXzqOVM5wv3jIX/T9aj79x5cPkEe7m5ZpyfCyjfenrODG4TMYvzC1UU4VSulnKokstBB8TLSbIRVW9p5iN/+sS20CKX/7pOL32fG86TJgss55ZOBNSQhg72YY1hWKJsL0ob5Y/GTpOYz7d7/MClmf/04Kd/zps3D9buZogQy7i8till22aY+law58NmMNb02KHgrrfzC/N2Ul01dup9TtZW0WrFbn77Zrtvtk3WxzHq1wclLpZ6tHQc89Ek3pmHE3WK0czRi9Vpw72kBdKt8AjLb3/tL4a6bqjasEbxLeUvj8elg3E8oOwE8WrC56wRvwza1szGvAdaWP4imsCfhyLDXuO9ZQcrUDpbHj988a9Hva0gav2rqPBesyJ+rop3820rivvjsu/Oqn+k02J5V+tlIcZ+GWxn3H8up4XySQsclZsQu9MXEZzR8bF1gwRI+9Je7A5C6j2N2HrXyIWfk8XLppD62fjDEhKgZ+91F7UcS537aH1drckNHXJC3XZE97+P4eqHEYT9Z4ka1UD+z7bq4vw6eRdL/BD7RF63dHdXlFizizCr3zZloEz9sxJutJ6Uv2lq7xLqX0M5BY3de/NGK8kM2Q+qJUOGDCUso8ksdjJLq66cMZnD0odvih1YxbsMGQ/ziZm6a/9vCM/ialfdAKGPHpL04iT7pXwsFlG/mu4MmE64jG6XnzoFoj6PU9O/IOiX9AFMo85W1wzuA/og5udn5pou52SG62b/jbVvDXWJfHn9Atk7h71BzbVueLR04q/VyJGAGDg7o2/t7pK7cbluO2j2cz/M/kJsUdKPVw+yezQ7Ifmnn9dXu8upPSMp6SvTy/6mpbqnZLB1z8DuRXidmWbo83plIOt6YTeci99ONi08dEI0TpRykzfeV2jn52Aj8vNDYD3Z+wbV9JfDedVajoHYXBgdz4mt5fxBK/ugGZ1u08wDPfL0qqE/uVjn9BctD3gT7z/ULd1AfdXvud5o/9GLE9WYy0d6I/W+ClyuhLEjvYAM+7r/WtjjVlYMxyFw/5ixaPG2+7RH7v8i2R2V3N1qNXPFod/hBmv/ESj7cm+96k9UKl+4//N+5gdjaglL6N2OFn9OseIz59K18AEv0lwS4BQ+W9PqXvyovdNYf/uUp3u55SicXC9bv5ZFqkmyK87ey0xh52foZzwyxb6n7N3ZNHnZ9A9UZwwp0xDYBUDIQ6zWQKDCPWkVbda/45Mnru07cmLU86x1ImkJNKP5fzrPt/mRnr3ZpY+cQrMZOHxq09JJx55q9haZhbZ8OuA4GBymj8vnQLj+mkq/Dz0Ffz+SAoj5PVk7TOc0zlVqc9E6XmN7ice51fky884CyAqvUS7gvfzFmLJyzqKJGm0FP68ep597flnJ7kKl5Gf7Zmc0SVqdjCPP3pckNXqDQMqR7fn7Ziu6FslrEeUtFkDrZE7HzICeF7aJjJ3hBedPmWvbSsV9XQsW7trnMGL+xh8NzeMHf0Fe/+zert++l+VD0KnHnGKtHh2R8W0aB6oU+UGFrTrLXZRqzkzfw3EpYrJkddQrt/Pi//3vEm3WJGJf7f6Hl0bVUnZFsiidkSeZi/pFnXkUn3yuuy+kVsbxSffrR0Gcu37KV21QKqFbrCJItO+EM0VeSkpZ8phv4L4xYnnWc9HL+lGWxxRlM2Znz6z/+wKCQBWDjJvD6bG4iNtPSNnju8nJWhg1Zb+LXZydiCxyytM0CTU8tXxwL6lv0fz248gZVb90XtC0bCcreGpStPZCDXqbNKl9H+cezzvwCwv9SjpR8vPzD8Ybw5SvqSePjXMrjv83m6+6Pp6a4DfqPnEHMpqb+YtTZi280jZ3LThzNM1WOWCmXpJ3Pb3v/5PAZc3j7w/a1JRdxxxhHJC6VDLGtShP21ivemrOQ9nXTUAt+tlYwlZeaBUW7pm/+FVhpOkbOkras8nzK+K4id3jhh6rSGleUhtveW3s633pPhz5X8WbSVSvmhbzz+nzlwQuh6DXaRjE8/mA7PTYi5/5K3/+LDG49L2IWybof+ZDW9XuAfEyranFg6iOA6U7HwTG5a+lFI5rb9anboU7n/+CWBSBM7B/nCq9YbyLXXvWO+7nCZzTSPf+A33kCuHqlIGx2rPQydXkoGuN6mgTAWTWKKKnVgc/l6s182fcGn8DX02sfuFvN6JbsOlEe86Ll3rJm1Hcn8tbsSNlYcUR5Oem0YvtZFppOTSj9WJ1q2aY+hqfLJnicZYvVTv4I3FLKJoMTtSWqRaP9ZklGoZo70+20LnOVd0/CpU+AijenTj3P+UreXEQMf4Py8vy2WChCO0AXQr/qMxQd3CZWPJPtsAn1g8K/LaP/Mz4E1a/NsWoTd6ud9tGbypskPbyUVyr3j9ng5a9DvnNaiNiNu6pR0ff6Olp4skAbKCDij/2TWJ+Hb9iuJpNw7Jg7eud9nFdaoXL5gd2Scvn59dlr6Vvj0N8/+nl573rNAGh1k0Cj2dd9Asy6wbFFksSSaaE+xeWPpRy2QYeveEmodVIBLx9K37rKFViSl/kPuorf+5KiG1WLWFG3Gux19LNX6o2Ipfe0p/feKbWmWJDYrt8aINQ+4d4wpoWQUvu90Pq9+cL+Mp8QXrk883tvvCih0OXB7vDhNuHnsNMKSduFtWUqjcddbI0wsbhgLjU+OXy4BVsTql1FYYjJvU6LEuz7Bu+eu2cncNbGXQI12e6V7UXMryE33Tpz9Vl23gBVsUX1+Hvm6PI1u+CCobpy+8L3F2JLLQzuPGQvn6mGhmRbN3Cj+84xfuInz3piiWyZadamYzh7Tpx9tx4Ed8M5JtsgTQu9f4ir8cPEzIdDNCjep74EfPTcP+CJzjObhidaVUuXdKXHbtyZ0Tip9MxSXeRjx16qEfHVW6RgznT5aGGbvETNZFhY9YOmMXAuid6av3B51JTA9/tUmdRlV5pZG70RpPdMPFo8bRl0FHpuTfvX5zdAyiBGD7DaJE86dn87h47//021Vu57VwfUKASf1+5VTXk5ykpeOsHb4+dskmK3VCBXKvaPH6xOX8fbk5VSv5OKiYxomVEeyFuZ/BnKZ+/Ero3D3zm9Lt0SWtcCECgQxJOXT9/29/N2pNKxRyVDZkG1h3zdEcVmFX4dE5xbcNWoO0eYQmY7UmPBEeZpku7htKtRtbe85ouBXeNGiXfwUbd7L49/+Q68TDrdFjnhX2t819iSZSE3vjXefRYEhwdixlrWfnLT0o66Rq2Nn+AcOo83AyzQOlHmQUppKrZwM/jZLKnon6FD/5Bc7sOo++X7een79d3PINv/YyJc6E2r8RDz8Z38Efw+xRqho3DkzQuHvK3Hj1sk0KqW0POKs6aPjaProOGsrTQBf21unKLftK+W696dFvJka7WORs4czh4pp6Ye99oXsSqDf2PmKrCfP0N9XGFou0VI50nSsufOEj3/4p5al6vzlHCuWwJhn7D3hnTOhVvOIzW2eGs+57epTr1phxL5MHIjM1FTofyzbygdhqcJ15zqEbZq+cnvANZmJxLX0hRAfCCE2CyH+Cdp2sBBighBimfa3prZdCCEGCyGKhBDzhRAdgo7ppZVfJoToZc/P0c4VZXuymfiuHmZDfHUCzFhlw8SeKJSHbCZj6SfX7kYPD7bC9pe6k7reybjGGrCVrwrsVfj31/9QV+H7GTs/ekqNYFKtb/Xa1ZqEgJCKX6Mna/hPSuX9mQhG3DsfAt3DtvUFJkopmwMTte8APYDm2r8+wNvge0gATwHHA52Ap/wPirQQo2/Eutf/Wh4a6ulXKmmJ0zeRTz+p82h/gxWq2Z9rpnxSbxRBF+Kvom1B25Oo1CSVKOaT/BdsPceJxYPZ7GyQ0LERb7YWyJMJxHvAh1vsieKfZGYlqb4GcZW+lPJ3IPzRdSEwQvs8ArgoaPtI6eNvoIYQoj7QDZggpdwupdwBTCDyQWI7AZ++gcHCTMeiFCZx8T9cknpLSlHjhr96J5OeIqG3E6+XAa53aOKwL39Kp+K3WE+thI/PRPeOFaTqd2Wy28YoiQ7k1pVSbgDQ/vpzrjYE1gSVW6tti7Y9AiFEHyHETCHEzC1bIiNSjBB1YoWO9olcLMP8+exclFmvZinNrZGbDIGzWBCyaTfhqZX9LFyf+Nq1Zmhd9C7n5E2PXzBBji1+m82k7wXZLqzoyr4B9hx9olmM1dE7umG4MbZHbpRyqJSyo5SyY+3atS0Vzmq27yvlvtFzKS5N/ZqsRtMwWHUiuyel7NxfyiVv/8X6JKJ77v9CPx3uJW+bS3mbCN0d0zlq6Vu21X9RlY/YRvWk68nEQVMrrPRlm/eytyQ9ETPFSU6KzJY1cjdpbhu0v/74trXAoUHlGgHrY2zPHLSGN3NT9B+/hK/nrIvIwGkpUfpDsE8/msxWZN8sD9OXdHl1su7SgvEw0qe/nbOOWf/t4F2L1h9IpXJrLVbxTv5r9p3g4f/YLWLnijGCJHfdO5B65alHNoR+J6r0xwD+CJxewHdB26/Xong6A7s098944GwhRE1tAPdsbZstxFN2VrsbUt3VJKlTagGfvvTlXom1tGA0sjExoeHonb1b+NTGgdvtd6+ASjVsq98qvrbT8EmQVLkVg3n4y/kpP6dZjIRsjgKmAi2FEGuFEL2BfsBZQohlwFnad4BxwAqgCBgG3A4gpdwOPAfM0P49q21LLWF9QG/6tCljIY3KLFWGrAi4d1LzY608i903/en9xuP57GpqCPOJyIzQuvgDKDC2zKQeRsILrSLaSlOpxOrVzRJh9Xbjs+vTRdzJWVLKq6Ls6qpTVgJ3RKnnA+ADU9JZTPg9YNVMQlsXUYmiuIwM5FrRAQPunRQN5GbAG7pBJLfuHUJesT0DtyeIj9kfZJOFN0tJmZdJSzZzRss6mCF72tc8evdhsg8Cs0dnwHMnLjmahiE9x6YKKSU2rUVh8Pz2lbey/e1cUeyGvPFc6ZxsS90tiz+kWBQA0Q2K6au2c+PwGSyKEZmUDX3ZSuyw9M0+I7OhyXNS6SdCqg2gZPunnQotmB1abqLkLH0TZS28EHa5d052LOBp10hb6m5Z/CEl5BtWYHuKy6LuS6V7xyh65xfCmrflbFC4mUCFUvrxOtbGXcUM/3NVAvUmIov5Y4JJ9c2blE/fxLF2Kep4CbAa9x1rqJ7GYgMjXf3iF0yAFsUjKCE/ZNsDX8yLmbrXrHWbie4dq2TSn9OSXOXm3Tsi7Q/WeFTMhGtR+L+RM/hnnfmJPHYOGPoXCg89X6hPPxVWf6oSrln5W4ITrlmRqrYq+3nf9SoOYf31PqfaV5QWR07xn7RkC9v2Rc/Fb0rBZKDCB1/m2PC1IKzi+bGLkzretHsnwxU+VDClH+8C+tMsW1WfnaS6cyWXcM1E2bBW/V1nnQAjCGHtw9iBl8GuN2jmMJbIzAzNi0fSpLoL0M/rEut35MJA4+QlW5i8JLHrnAuk+u0rJ907iXTsFHkvAiR788VcR9cGQhOumfvByTwwvp27LuFjkyH8+jzsHMUZedaHJTYvHkmZAdsrWhtGpgaP3dbhu5ekOJeMnW+ldkTRmX6oWnjuVTbd4zmp9OMRrW8k2mdSNQ7gZ/mWvcxfm/ji44mRuMBPf78oLbMlgxXM1cP+5o5PZxs+Nljcno7fucVpzOdvhhbFIwIKP7x5jCuPxNVMqdubFXHl6cRsr529eifLN1ujrK//wJ5w4Aql9O3SOxt36y/fZxcbw5YLTMUre8h6o6mMk7BofsD8tbsM55iH8jY9RixjYP47iQsRhRbFIyjFFXW/22Au6+Br7/FKXojhw96wqziifC6RKb/GqrQsZTqrn1lBTip9I0opYj1VKTMid4cRUqp0NWTIZ/PtZLRp7V6420xulHpsY1j+AIsliK/wwfhi28E94felW3hvSvS88QfKPGze7Rs3uOTtv1i8MTXZR1NGeN+xoPNk4BBI0uSk0s8GkrHO05FTJEueh3Fp+7SxlE8F3hKG5g+klrBWMRpR+OBT0EYIDtkMt9z1Fg7ZHhQJNGraakPnsBI730rTcV9kIxVK6cugv3pKLFu6TDoUcLI3VMonv2n/RchhSBDJ07xNO0d0qzkRHmw5wZDCh8RCTM0ekS393SgR4yJZYqan+mGVk0o/6iIqJiMb4pGqMMZ4pNqnb+t5LK0rsdpuz/uOHvxpoSQ+C9+Tlx+/oEnMrrwObKEAACAASURBVKBWkazhXHk7tZqcVPpG0J29lyU3RDqkTPYGMvyAjPDLpvbXnuWYyUOuzy2t88UOv1GKK+GxGEn0ax5cp5Hac1kR5vJvs5IKq/TDiebysYukksJZJ4Zhgh+ItrZTGhOutRSrecP1hnUCADy+GbfwWfh2v5EZuSzZ4vJIhPDfny2/VU3OsoB41zqbInUyhaQtfcsLGqnKeGU12c17rgEUCnOzsmPRongEOAsCeYti9ctEf3a2KLZ0kG23eKpCaHNS6UcjXpOmso+kKmulHWSLojHaxk7cvJ3/Ooc6rEsFYDRKxwixfkfwtTByWdIdmm9n10nVQj92YVdcfjg5qfSjZh6U5fv1M/KZO092dzFzBN9Q2XJvGX04Pe0cQWdHcom5gglX+Kl6qzTk3rFdCkWiKKVvB1qPj34Tpk6bZYu17Cf5gVyD5ay6BsLYOa/Nm8C1zonWnBPoVfpwhIUvw/5aiZEV1BTZgVsno64dVCylHzyz3YbZe6ki/KGV6amV03XeeMee4FjI867hSZwhlLmVTuA3b/tIOWTo30SIdqxZnZ9N/dws2f7bPCn6ATmZWjnafRDPikxlp8m2DpqsiyLTwmEPE5sY4nrdsvo+cp/JpLp9YcfmiH3+355oG1jadkE3Rzr6oL0vJpnVx4wSeBNMkfgVy9IPQu9Gys4ukxpS1TYp6fjFu3nP9So1hTULdwwou5Qn3DdF3W+naywdeZgylWwzpMJJlWGUk5Z+PKJ69FPYa5LLvRPKvLU7k5LF0DmDXWNJHp8q9K6nAy98fTMtHNbk6X+krDejPF1jyxGQx5JThmC6H2W5YoxFuiOTkiZF8uek0o+ehkH/M2SPItNjT7HxzJGJk54fa3UbP+gcDUt/sqSuW0rvZby3U+B7NKPByG9I1OBIxs7PNJdbsoT/nmz5dYExnxSdLyeVfqJkS5x+Ogi2oux0KNjZLBc7/uA25/eW1HVl6eP87W0dsi2a7FOKfPH/iSr2WEdlW/CO2YXczZBt91Q4qZK/Qip9KaHEHRkTmy2dJj2ukqDPltYr+Wv5Nop1rkfSdQd97iCWMij/bUvqPbfkRRbKxobKLt20hzXbD0TIY5boVrk5JZpr1n0wEWkY0iJF4iiffhJEsyaCFVf/8f+G7TOfmiGZS5RtFlrS0TtRDh8zbz33fDY3+nEJni+4eRuJzXxd8HSCNYVyWslA/pP1dPfp/cbdB6xJ62DLgz7H9H94H3194rI0SZIYKnrHZjbsjFzi0GybZ5neTorgtlm+2ZqoF4C1Ow6EnsfCni+lL6fOlIJ7LamvU/FbURW+UXnScawd9WQiGTFQngASybQV25RP3w5CMkXq7Dc7IJrU63qW3XzB8k5bud388Qm21vfz1id0HEA19jGn8NaEjw+mXfFQdnNQUnVk2SXPOuxwj6TiPh09Yw2v/LTE/hNpJGXpCyH+J4RYKIT4RwgxSghRKIRoIoSYJoRYJoQYLYQvr6wQokD7XqTtb2zFD0iUbE/OlGqyLSupq2Q78wtvtqSuI4s/MKTw47VQog+w2K7HxK9Ldl3R+GRZFw3w37b9KT1fwkpfCNEQuBvoKKU8CsgDrgReBgZJKZsDO4De2iG9gR1SyiOAQVq5lBLcKdId05t1Pv1kjzeYRsCKy1KHHZz09fEW1AQtiz/kAIWGymbbg3HFFuvcdEaxNfIrS907qSZZn74TqCSEcAKVgQ1AF+BLbf8I4CLt84Xad7T9XYWd8Vtx2FeSitj23MGOG2rMvPWMnb/B0jobiS1ML7zDkrpaFI+gBONLHCbTneM1r1XNH1zPvLW7LKo1M7Dj7T3LnuOGSFjpSynXAa8Cq/Ep+13ALGCnlNKvUdcCDbXPDYE12rFurfwh4fUKIfoIIWYKIWZu2WJdfvNwZv23w7a6jZBtnWnNjuReQfV+7t2j5rBw/e6k6g2mqVjPz/kPWVJXJ/Gp6Xz4SVn6MQ6VMnp/ybZ+ZCeqKYyRjHunJj7rvQnQAKgC9NAp6r8WemZQxHWSUg6VUnaUUnasXbt2ouLpojpF4jzy9YKUnCdRJdZKrOar/KepLEqSlqFF8QjKhDULoBgmzktCLsfXW0YSTTRj1XbmrYlMZ5KL7p1konfOBFZKKbcACCG+Bk4EagghnJo13wjwj16tBQ4F1mruoOqA+TCQDOK/bfsSPjYXO1Ms7PR3txdFfFfwpCV1+RdAqWJJbfYzY9UOmtetarh82scdbOz3ibp3/tu2j8vemaq7761Jy5MRKSNJxqe/GugshKis+ea7AouAScClWplewHfa5zHad7T9v8q098ByPpuxxvQx83PMJ5qNdBKLLVf4YF26gH4//hu/kAGi3SmPflP+BpaOgdlMIlFl8vSYhZbKkekk49Ofhm9AdjawQKtrKPAwcJ8Qogifz/597ZD3gUO07fcBfZOQO0GZo+8rsnDCkRE+nbY6pedLN3Y83U9zzOPzgucsqSt8icPt+0pN16EXHDDT6NhRnAYy0n4vWfSAsROrZijrkTkmZGaT1OQsKeVTwFNhm1cAnXTKFgOXJXO+5FG9Ilfo5pjOW67BltRl1SLms1fbl+I6nkJbs93YQHu674BR082/URslUffOgTKPxZJkNhUsDUMFc6RnEFZaYRc5pvCWazBOkXyStpbFH1qi8JMmbteM3oArt+7j4iF/GTpNLlvDif60v1dk9dCiaSpUGob02zmKxJCsKryGKZ42jPN25nnnB+ylEtVILoy0VfFwU3H4tpJgyCbAGa9OtlycrCSXn2gWUqGUvuoTaSTBts+njKWFvvH/k/MWcnLeQv70tOGkvOQG31oXf0AxBUnVYSWxmkeqgE1DpHuWfbZQwdw7imwiD09A4fv5wdM5aYXfrngY+w2mVkgVyvGYPBkUDJjRVChLX5E+zNqqAi9DXQNDtn3uPo3tVEtKjmOK32F31kTh+/C5d5RCi4fewkiKSCqUpa9um2xB8oxzBF3z5gS2DHd34yPPWdyaxHKHxxUPYUeSD410ofpufFYbjGCq6FQoS19ZS+nDTNM/6BzN9c4Jge9vuS/gDffF/Ft4Y8LnP7F4MFuokfDxdhPbp68wgtujWsoIFUrpKzKf2/LGcIdzTOD7YPdFDHRfxqrCaxKu87SSgaynlhXipQVfPv10S5H5qOFuYyilr0gJRm7Ha/Mm8LDrs8D3Hz3HMdB9OfMK/i/h855V8kpSSxymingDueotNT6qiYyhfPqKjOBixx887xoesu22sv9xR963VBeJ+WovL3mCZbKRFeLZTjz3juq78VFtZIwKpfQV6SOWpdrNMYNB+W+HbGtc/CkdxFIedH2e0Pk2N+rGdHlkQsemg1jtI5XWN4SdeX1yCaX0FWnlZMcC3s0fFLKtXfEwarOTrwueTrjeRSe8mqRkimzj50Wb0i1CVlChlP7kJfatxKWIjZ6h2kEs5eP8l0K2/V/p/eyngBmFtyd8rsfKbsLjyJzZtsmjhigV1lGhlL4ic2gjVkVY8gu8jfnFeyxFhdcnXO8ab20+95yenHBpIFb+/hfGLmavWtNZYRFK6StSQrDLuplYx9iCRyPKnF/6AlML7kyo/qme1gC87ulJGc6sW5kslk9/knpDVViIUvqKlND3q/kANBJbmFjwYMT+44qH8JRzJPWF+TS3g90XcbDYzXJvfb7xnAyo8D2FIhpK6StSwsR/N1ObHUwpuCdi332lt3Ja3jxudI43Xe8I91kUeRvR0rGWQe5L8ZBnhbgpx6rlGRWKeKjJWYqUUJ29zCi8I2L7ZlmDPVRmWFhyNSP84DmeZ93XMyH/QRZ7D2Ws93grRE0LavKVIlUoS19hO1U4wLzCPrr7Xiq7imH55hX+DG8L7iy7m555f9DUsZGB7suQQd05kTVu04lS+YpUoZS+wlYKKGVhYe/A96HucwOfX3f3jJiUZYRNsgaXlz6JCw/3OL9mnrcpE7zHhpR58Mv5iQutUOQwSukrbGWga0jI9z7OsQBslDW5x/l1QnWeWPIGEgdX5E2ikdjKAPdlqGVIFApjKKWvsJUzHbP51N2F00pCXTj1xI6E6mtRPAIPeRRQyp3Ob5nubcnv3nZWiJpW1CNLEY5dfUIpfYWtlJBPCS5+zH8k6bpaFQ+nFBfgy8hZT+xgQNnl5ILKVD59RTh29QkVvaOwlf0UJBSKGc5Rxe8FFjI/RizjHuc3/OE5imlZlFRNocgElKWvsJWa7Em6jg7F77CXygCc5FjAx/kvsl1WpW/ZzUnXrVBkKsq9o8g6BF4KRHI5Y04vGRBYDP1sxww+cPVntazDZaVPsY7aVoiZEagwfUU4dnUJpfQVtjEp//6kjr+85AlWyfoA9HT8zhDX6yyUjbmy9ImMXu9WochklNJX2MKFjik0diSe3/yBslsCi6D0yhvPwPx3+Nt7JNeWPsouDrJKTIWiwqEGchWWU4tdvJ4/JH7BKAx1n8uXntMAyZ153/KA6wvGezpyd9mdlJBvnaAKRQUkKUtfCFFDCPGlEOJfIcRiIcQJQoiDhRAThBDLtL81tbJCCDFYCFEkhJgvhOhgzU9QZBaSmYW3JXz0dG9LXnRfA0gedX7KA64v+MpzCreX3ZPTCl/lW1OkimTdO68DP0kpWwHtgcVAX2CilLI5MFH7DtADaK796wOYn3+vyHi+yn864WNLpIsrSp/AgZd+zmH0cY5luLsbD5TdkrXZM42iBnIVqSJhpS+EqAacCrwPIKUslVLuBC4ERmjFRgAXaZ8vBEZKH38DNYQQ9ROWXJFxnOWYybGOZQkff0zJuzjxMtj1Blc6J/O6+2KecV8fkkhNoVAkRzJ3U1NgCzBcCDFHCPGeEKIKUFdKuQFA+1tHK98QWBN0/FptWwhCiD5CiJlCiJlbtqgVg7KFauxNKFumnxOK38CLYJhrAOflTeP5smsYpHLqKCowmRin7wQ6AG9LKY8B9lHuytFD7zdEvNRKKYdKKTtKKTvWrp07cdi5zvwoqZONcFHJs+ylEiPz+3GKYz4Pld3Me55z4x+YQ6ilzxXhZGKc/lpgrZRymvb9S3wPgU1+t432d3NQ+UODjm8ErE/i/IoMYZjr1YSPvbv0TtbI2ozKf56jRRF3ld3F554zLJROoVAEk7DSl1JuBNYIIVpqm7oCi4AxQC9tWy/gO+3zGOB6LYqnM7DL7wZSZC8nOxZwVt7shI4d7L6IGd6WfJ7/LM3Eem4ue4Bx3s4WS6hQZCd2uXeSjdO/C/hECJEPrABuxPcg+VwI0RtYDVymlR0HnAMUAfu1soospgoH+Dj/pYSOHefpxLeek/mi4BmqsY/rSvsyU7ayWEKFInvJyCybUsq5QEedXV11ykogcpFURdYSvCKWGf7xNuY99zmMzn8WB5KrSh9noWxisXTZxfa9mbO849Pnt+bp7xelW4wKz4ZdxbbUq2LhFAnR3/lOQsdtkdUZ4r6A4fmvUIaTy0ufDFH4Ai835f3IANcQKlKW+X2lnnSLEEComWI5jUrDoDDNsWIJlzl/N31cmcxjuLs7r7reZaOsyXWlj4RkyqzPNga43ubEvEWM83RChWumB4dq9pxGKX2FKQoo5auCZxI69kdvJ+5xfsUK2YDrSh9hK9UD+y5w/MnzruFUE/v50H02z7mvs0pkhVmUpZ/TKKWvMMWSwhsSOm6NtzbnOv5mrjyCG0sfZLeWKbMae3neNZwL8qbikYKnynoxwtPNQokVZlEqP7dRSl9hCAdeFhckHnB1qGMLf3iO4pay+9hPIQAnOv5hgOsd6ovt7JGVuKvsTiZ7j7FKZEWCOJSln9Mopa+IS1uxgu8LHk+qjh89x3FP2Z2U4qKAUh5yjqa380cA1spa9C59gCXyMCvEVSSJ0vm5jVL6ipjUYxuf5z+bVB1fuE+lr/tmPOTRWqziNddbtHCsA2CO9wj6lN6nVsLKIJTOz22U0lfEQPKyaxiVROIx5MPd3XjWfR0CuDVvDPc5vyBf+MITf/B05v6yW3M6T342otw7uY1S+oqoXJk3idPy5id8/GvunrzmvoRGYisDXG9zvOPfwL7B7osY5L5UpU3ORJTOz2mU0lfo0khsoZ/rvYSPf67sWt739ODSvN95yjmSquIAACXSSd+ym/nGe4pVoiosRln61lMlPy9jJuAppa+IQOBlgCvxhc0eLOvDL54OvO16jR55M3BLnzW/XR7ELaX3MUPl2MlolMq3nt6nNGXO6h38sWxrukXJXaXfsm5Vlmzak24xspLr8iaEuGLMcFvpPRyggPEFfanOXpZ4G9FMrKfI24Cbyh5ktaxrsbQKq1GGvj00qF4p3SIAOaz0FYlxuNjIs64R8QvqcEvp/zjJ8Q/XOyewxNuIBbIJXfPmMMXThtvL7glMyFJkNsq9YwMZtAiyUvqKAA68jMlPLB7/mbLreMj5GU3ERka6z6KB2MqZeXP41N2FJ9034FZdLWtQOt96MkflqyybiiBuzhtLdbHf9HE/eY7jMecnFIpS7i27nXaO5XRxzOW5smt41N1bKfwsQ2XZtIdMWRIzZ+9G1W/N0Uys4xHXKNPH7ZRV6J43g288J/GZuwsD84dQg730KbuPX7zH2iCpwm7UrZPb5KzSzyAXWsaTh4eJBQ8mdKxAcmfpXeyngPfz+7OHylxe+hQLZWNrhVSkDGUwWY+UmaOTlHtHwfLCxNIYT/G0oVvJy9QROxnmGsBKWY8LS55TCj/LUQO51pMprh3IYUtfYYzBrjcSOu6Zsuv42HMWTzpHcp3zF8Z7OnJv2e0c0DJoKrIXpfJzG6X0KzBX5v3KBXlTTR2z3FufW8r+xyZ5MO+7+nNq3gLecZ/Py+4rVEqFHEEZ+vaQKba+UvoVlPMcU02nWRjmPodX3FdSV2znq/ynaCI28lDZzXzuOcMmKRXpQEXvWE8m+fSV0q+AnOP4mzfzzbl1rix9nL+9rekgljI0fyBOPFxf1pep3jY2SalIF0rlW0+G6Hsgh5W+Mlb06e6YzpD8waaO6Vz8Bhs5hAscf9LfNZT18mBuKnuIlbK+TVIq0omy9HObnFX6mfIqlUl0c8zgnfzXTB83qeD+QE79ad5W3FL6P3ZS1WrxFBmCUvn2kCkRPGrkrYJwpmMW7+YPMlx+tbc2jYs/4YqSJ0IWUbm29FGl8HMcZehbTyYZoTmr9FXHLaeLYzbv5Q8wXH6jrMmppa9Tm52MLngusL1l8YeU5e7LoUJhG5li5UMOK32Fj9MdcxjmMq7wATqXvMXRoogZhXcAsNJblybFH6tlDRWKHEAp/RzmNMc83nUNIk8YtzKaFn/MZXmT+bbgSQA+dZ/BGaWDVAx+BUK9JdtEhhj7Sd/JQog8IcQcIcQP2vcmQohpQohlQojRQoh8bXuB9r1I29842XMronOKYz5DXQMpEG7DxxxZ/AHPOYfT3zUUgIfLbuZR9812iahQVBxkxuh8S8y3e4DFQd9fBgZJKZsDO4De2vbewA4p5RHAIK2cwgZOcixgmGsABaLM8DHHF7/JD/mPcY1zIgCXlDzFaDXpqkIiVPyO5WSKwocklb4QohFwLvCe9l0AXYAvtSIjgIu0zxdq39H2dxUqINhyTnAs5D3XAApNKPyLS55hWuGdNHNsAODE4sHMki3tElGhqHDIDArfSdbSfw14CPBq3w8Bdkop/T6FtUBD7XNDYA2Atn+XVl5hEZ0di/jA1T8kxDIej5T15puCpwLf2xS/z3pq2SGeIltQppgtZIriT1jpCyHOAzZLKWcFb9YpKg3sC663jxBiphBi5pYtWxIVr8LRSSw2rfDHezrykut9AFZ469Gs+CP2kRmLNysUCntIxtI/CbhACLEK+AyfW+c1oIYQwh/M3QhYr31eCxwKoO2vDmwPr1RKOVRK2VFK2bF27dpJiJcYJx+RfVZuR/Evw/NfobIoMXVct7yZAHzmPp0upQPwkGeHeApFhSdDjHwgCaUvpXxEStlIStkYuBL4VUp5DTAJuFQr1gv4Tvs8RvuOtv9XmSnvO0EcenB2WbodxFI+zH8FZ8DDZo5Hy3rT190H9U6vUPioUdlleZ2SzBnMtSP4+mHgPiFEET6f/fva9veBQ7Tt9wF9bTi3BWSP8jtGLGNE/stIhKlIHT9XlDzBp56uNkimUGQvV3U6LN0i2Iolc+qllJOBydrnFUAnnTLFwGVWnM9OHGnW+fWqFbJxd7GBkpKh+QPx4KCG2Gf6PKeUDGKNrGteQEXOkz1mjz1knv/BWlQilTDSHURq9KFTm53UFrsSOke74qHs5qCEjlUoch078uRk0iIqam59GOleFNro1IVWjjWm617prUvz4pFK4StikiG6KX3Y0AAq4VoGkwmvtp/efHzM/fmU8VF+P1N1fuE+lTNKB6osmQpFHLw2meSZovaV0g8j3ZOEhYATm9Xi2s76g0l5eFha2Et3XzSeLOvFg+5byYxHmiLTqei9JFPcMHahzL4w0u3Tj3V+gZflhdeZqu/a0keY4m2bpFTxMT4ArVBkNnbo/Ex6kChLP4y0+/Sj2lmSlYXXmqqrR8lLKVH4kP6oJzPUOkitCxCLDNJPacFKBd2oZvm8n0yZlqSUfhjp1l36ylOyqvAaU/V0LenPYnm4JTLlGuPuOSXdImQ06b4H0o2VPv3eJzcJfM4Mla+UfgTpdu/ovWmYVfgXlTzLctkwfkELSfdYiBnqVC1MtwgZTTZdS4APbzwu3SJkFUrph5Fu9064mbWq8GpTh99deidz5REWCqRQVCzscMME11ngTK/azXmlX7XQ3Fh1uq2c4LMXFZjz4Q93d2OM90RrBVIoKhhWqnz//SytrjgJcl7pe73mWjrthr4mwEMLL8IpjCdRW+WtyzNuc6GcVuLI+Z5Uccgu5471WGno++/n4DqPb5reZURy/lZt16hGzP1dWtUJ+Z7uDu8QwPBzqFa21dRxp5cOtEcggzjTrPWr5EdPC33zKU2oXbUghdJkN868dN8F6cXu2bOXHdvI1vrjkfNK/5FzWlG1ILqL54MbQgeB7PLpt6xb1VC5AY7B8N+f5uou/pB0P66Cz76q37lRy9kV2rnw2e6ckGYLKpXcclpTW+p95ZJ2uPKsUwt3d22edB3ntqsfc3/nBK57rNvcpHPAMP6HSbq9CTmr9E9v6bPg61Yr5Mj61Qwfl+gFufmUJjH316oaPzZ8QccfabvjF1PnPan4dUqwP+48liIHKHHHd0Vd2/kwljzfw9D5zmhZvoBONYPjMtGuXfCr9fRHfamkX7/yaEN16tHjqHoJHxuO0d8Wjlm3pVHydQYZZz9xlm7Zww6uHLe+m05qnKxI3HJq9AfclIfPoNCVx+Jnu5uq85EerQKf61cPjeay1r2j1Rn09pDuhedzVuk/2K0lfz/SlbrVoofnndoicmWu4MkUsbikg/FXtHyng0qu2Df31/lPUvWfjwzXCXC38wnWob+62BUdDzVUx+GHVA5RsMGceWQd3e16FJd54pZ5qHsrw1Zk8BtXq3rGH9p6eHTu4hObmVsh7cnzWgc+x+pTZil0JbZamV3WqB7RVFQLA2+vNSonb5DEUpINa/ju10o67r17ujanc9ODI7a/c+2xHHZwFQDObl03YCAGzmehTg4M5CZwvTocFts1nSg5q/TzHIJ61WPfnLed1ixi2+UGleXFx4TGwceK+nn6/DYx6+ogltLBUWTovH5eLbuMaY7o1mq/S9qy4sVz4tZzcJV83BZokMa1qoR8r14pcvUhM/dScHPGmywz+KpjIo4JRu8txKzf9sKjG5gqbwfTH+1K24bVAfAkeM2a1qrCqn7n8mOMCWrhzR3crscEKSKjrrqRN0Usr2HqTStmapIYO6/tfDhvXt0h8GDwc0SdKgHjrm3D6uSHjWHkW+jeCsaM4v/twdMZfcsJtsiRs0o/hCj9Qs8K0OtErcPcQ7ed3oyTm4dairH6f7yBsd7OcTH3hzPN24o3PRfHLCOEwOEQHBQ2nlHoCr3kFx3dkFIDrplgTmwW34f6V98uujLF48Mbj2Pek2cT3KIXhT1g/3jojJDvTbUHTjSLsKRMR+lrN2Ctg0IHeKO9wQXLno446/d7daRO0BuGmVmjdaoW8MNdJ4dsO7J+tYgghmg4grR78LiJngQ1dZYa1Huj7tbGnIts+mNdmfpIl5AB+3gPYq+U1DqogD467qGjGlbnp3tP4Y4zjoh4+9RzcVmJkTeJgwqclo6tBFMhlP69YYNJV3U6lEXPdouqhIItmEY1K+EK6gQHFTh5uHuryIO0Y244sXHELmcMk6iJ2MC5edOjC6/DFaVP8On/xU6/7KfME6rw8oJ+c9ELPbj+hMMjysTjzi6Rk7/CJ7RUKXDy7IVtaFG3PHe/EcOwaqGT6pVdITdGl1Z1QsYUHCZHg0s9XprV9j0Y/DeSX9xE7qsbTmrMRSm2/MPfnPSUfvgDzE/96oUB90fwUUYnIbkcjgirPlpggtF5LmYDJupULaR+9UpMe+xMnrkg9puzn2hvsP43v1b1quFwCJxhncBlZfSSP2QTc5Z+om4/I1QIpX/iEbVo16h64LvHK6mcH93HHqzUpQx9CHRsXFP3GL+VqRcamBdVSUkmFdwfQ/JI2ns+AgRHG/T3hbsBgr858xwIISg1qfT1TDy9/nz9CY25p2uLwPdY93k9zYr13xjBReNZXn43XlT3TpmHd6/tyIc3HkfNKj4fs19phr8dGNFFBc48HtR78NtIuFx6l2zm42fqHlslRvSaHuFvg648EVDmfjmOa6J/HxglUb/5QQXOwD1WOUaYLoDHo69li8Pe/MLdO0Ys7Nb1qxl68Ov59PV+evtDQ+9nO98mK4TSh1DF2zZO7H6wnnzl0na4gmLQo1ntsTqxficyn0TtrwunUGoyUid8EFNPzEY1okVh6P8of53Bbq9oVkxwu/gV7Fmt6zLoivasfMk35tC2YXUahg2gB0eGhCt9B/XZLgAAEL5JREFUf+RKwxqVWPJ894CFG+1GObddfapXdoUM2PnFDb9uZnVRgzjjRkYwcoOHK12P16e4wt13fpwOwdRHuvBQ95a8ZjJSqV2jGvTrWZ6dNfjeOaqBz3jq3PSQiGv+zrXHBjKYvnd9x5A6wskzofXDi3ZrU4/7z2rBo+ccqVv+2MNrhhwX/kZTEhZ04L/fa2iuKSO5mR45pxWvXXkMN50UO2ovWPYHurWgXaPqIa5hf9s+eV7obwl/+7CSCqP0/Yr75Uvacu3x+guU/Kn5oYNfnU86ohYuZ/mVi/ZaennHQ6la4OSC9pFPf6dDBDqinwUF/xdf6Ms+BGC5tz5HFn/Aice0ZdAVR3Nk/WoUOqNbOS9cfFTgc/iNqSd//8va8d71HUO2Vc7Po28PfWvWbznmOx28fU0HJvzvVI6o43Pj/HLfqWHnK//sP/Ww6zty8TGNEEKwqt+5fB/mbwZfpI+f8IE1/29yOHxWt58Xe7blltOaMuXhcp//2a3rcuHRkcnn/IrAiJvhoAJnRDoPK/OzRBvUbBI0OO5/MLx8STvOal038MCN5tcuevEc6levxO2nH0GdqoWmH2ZXdiq/R4QQXKqNdXRuegiznziL89o1CAl1bNuwOt2PqseImzrxUs+2nNm6bkgdwZzbtj4Oh4gbBhyNPIfgrq7NqVoYOX5wQfsGvHPtsbzUsy2HaobD4Vo7+l189cMGdv1vujef0pRXLm3HlcfFD+Y4+Qif4o4XEHBe2wYcc1gNbj+9GUfUqcqYO08Okdt/fzSrfRBvX9Mh7nmtoMIsouJ/ojaqWTmq39E/yt+tTT36j18S2H7vmS34s2gqEH1QtkmtKix4ppvuPmee4JZTm9KtTV26DPiN+5yfU1UciCGtgKs+g+qNoHk3rlpwAQfw3WDdj6pH9zhx4tccX55SuVW9qvy7cU/M8lULXZzZum7gu5mbsUdb38SZ5y48igvaN+CIOqG+3vDv8fDfQvlOB/16tuWlH/8NKLxZj5+J0+Fgx/5SIFJh16layCM9jmSnth/ix+4bMTj/0a5rnkPg8UpDCnTg5e257/N5cctJ9F01AJMeOJ3GfccC5W+LR9avxrDrO1Jc5mHZ5r081K0Vizbspn2ct9dkeeScVlx3wuEB9xjAY+ceiUPAiKn/Be6v+tUrcVUUZQ+++SyPnds66n49jMa1L3uhB3laAEOwDGe0rMOYO0+ibcPqbN9XyiFhYx/+QIZCV14geu+ZC9rw1JiF0WXSSa8Qzle3nUj1yi6+uf2kqGV8fVgiEPRoW5/P+nRm1n874v3UpKgwlv5LPdtyTtt6UX3ywfitVj/HNS6P8jFiGZ6mRSucor3G5TkcOByCprUPYsQxy7jb+W3sCnq8Ai27Q72j4JrP2Uzi/tNRN3cOfH7snCMDSvWxKK/GiVIpP083SiO8LaOh5/u8stNhzHvq7MANdshBBVSv7Aq4l6Jdi+CB3mgKI+DTFz63hJ9YeVFiDciH09PEPA69eQTxzl3oyuOFi9sGlMrTBgc3g99Quh5ZN0bJUGpUzueohtVDthW68jhHe+jHG/y8/yzf2E742KqR0E2jniBXniPqIH+7RjUQQkQofICrOh1GJVdeiDF1epS5K+HEiqIKHxvRI3y8r3PTQ7jjDHuz5FYYpd+4VhWGXHNsiDsgHt2DwsoGXdEeiDUoW857vTqy6NlugagYl/+YRd9x2uKnYh98/G1wfB/DMsYj2DK7OSh07YpOxuYjDLisPZ+ERQqZ9Wz4LfVYx/nlNBIuV+6a0d8f7C+ONhO6Uc3KnNO2Hm9e1SHkZr9Uy4uil8tHb2wm9vyM1vx4zynMe+psFjx9dtRyRmbXJhu+pyfnNccfxvwwuZrWrhJRLhb+QIF4uZcuObYRNSq7It4CDqkSPSfSoQdrE69sjGQBaFmvKouf6x4Sz6/nU9e791vWi/4ma+Q+CRguKZykW2HcO7Hof2m7kEknEOniaFDd1yHaNCgfvPyzbxdO6vdrRH2uPAeuPAcPdmvJ3aPm0s4/Mr9lCdRrBxvn6wvSogd0eyFic8u6VVmyKbaLBuDq4w+LmDQGvtmkDbQObbZvXRIjOZRRC+yr205kzLz1MS2f/pe247u562nfqHrUMn78OjKapR98c0Yb7MtzCIZcc6zuvhmPnUm+00H7Z34O2e537UnKH04Na1Ri3U6fq86/TnC3Nj4L+oawQb7DDq6MQ8CqbfsD2wShUSjh1/qNq47hwS/n2ZIwTghBtSD/8l99uwT6Cfgs4G/mrI1ZR5lf6cex9BvUqMTcJyMffCcdcQhPnNea535YFLHvq1tPZNZ/OyIm/qWChjUq0a9nWxxC8NBXvvv1z4e70OP139mxvyxQ7upOh9G+UQ1Wbt1HgdPB17PX8dPCjUBkegc9Al01hTOsldIHLjMwC/f4pofw/Z0nhyj9hjUqcf9ZLfh75TbdY449/ODA4DAApz0EK38PKXNr6b3M8R7BtB4bofNt4Ii0akbf0pmVW/fFlfH/Tm5C09qR7pSbTm6iU9o8rjxBk1pVaFWvKrUOKuDBs1saOu6ohtUjXAPh1KicTy+dOQ56eOO5d4K2xwrNjUY0Beu3Zt0eL3WqFfLW1R04odkhzF2zg+/nbWDQFUezcVcxNatEDjAC/K5NKvP76cF3r5/Vui4vXtyW89vXp5IrjyMe+zGw//z2DThfJzjALHW039Tn1MhZ6C9cfBRrth8IUfjgc4m+FCMCB6BM84cn+iYihKD3yU1ClP6tpzULBBH4x4zSgX8gukW9qrRvVB0hBHPCHlxCiJD+fXabekgp8XiloQgcvzvKiIvPKpTSN0FbHSv0rq7NuQsTmQRrNYdVf0CL7nD5SH56XEuwdtp1UQ+pUTmfYw6LdFOceWRdPpm2GoB2jarrKvxoJNLHljzXAyF8HT1aTHgqqKuF1F1yrP6SkEZccOHc07U5r09cFrLtwxuPY/X2cqvc77f2W7f+7I9dWtWlSyufdR8v9QfAl7eegMMh6DnkL8DXnlcHRZT98dAZ7DpQFu3whKhS4Iw6QB888G8WtxY6ama8Q4/Hzz2SE5odQoEzj8aHxE/klkqOPtTcILkQwnB6ar8rMtG0GomglH6q6fIENDoO2l2ha9Wb4ZkL2nBZx0O56K0/DaeXHX3LCXwzZ61udseOh9fUHYz1Y3YmrF3UrJLP0ud7RB08dAhf9soHuxl7EwH431kt+N9ZLUK2hSfi6tTkYL6buz7piTMdGx/M5t3FUfcfenBljI24pB9/Mrxk30b+7xR7UkVnOt2Pqscn01brJoyzC2HHepBW0bFjRzlz5sx0i2Er/lf9RGOWAVZv20+DGoW2TuhQ+DKJLt+ylzYN4o87xGPXgTLaP/Mz3dvU453r9McWEqFx37HkOx0sNZjC2go8XpnQ21VFZvW2/RS4HBxSJZ/t+0sNTQgzgxBilpSyo96+hC19IcShwEigHuAFhkopXxdCHAyMBhoDq4DLpZQ7hC984HXgHGA/cIOUcnai588Vvrz1BFZsie+vj8VhGfY6nKsUuvIsUfjgy6Uz7u5TQiZgWcFLPdvSqUlkIkE7UQrfPMH3rNUKPx4JW/pCiPpAfSnlbCFEVWAWcBFwA7BdStlPCNEXqCmlfFgIcQ5wFz6lfzzwupQyZtawimDpKxQKhdXEsvQT9gdIKTf4LXUp5R5gMdAQuBAYoRUbge9BgLZ9pPTxN1BDe3AoFAqFIkVY4gQWQjQGjgGmAXWllBvA92AA/KNhDYE1QYet1baF19VHCDFTCDFzy5YtVoinUCgUCo2klb4Q4iDgK+BeKeXuWEV1tkX4lqSUQ6WUHaWUHWvXNjYVWqFQKBTGSErpCyFc+BT+J1LKr7XNm/xuG+3vZm37WgiJRGsErE/m/AqFQqEwR8JKX4vGeR9YLKUcGLRrDNBL+9wL+C5o+/XCR2dgl98NpFAoFIrUkMzkrJOA64AFQoi52rZHgX7A50KI3sBq4DJt3zh8kTtF+EI2b0zi3AqFQqFIgISVvpRyCtHzd3XVKS+BOxI9n0KhUCiSR03hVCgUigpERqdhEEJsAf5LoopawFaLxLETJae1KDmtRclpPXbLeriUUjf8MaOVfrIIIWZGm5WWSSg5rUXJaS1KTutJp6zKvaNQKBQVCKX0FQqFogKR60p/aLoFMIiS01qUnNai5LSetMma0z59hUKhUISS65a+QqFQKIJQSl+hUCgqEDmp9IUQ3YUQS4QQRdpCLumU5VAhxCQhxGIhxEIhxD3a9qeFEOuEEHO1f+cEHfOIJvsSIUS3FMq6SgixQJNnprbtYCHEBCHEMu1vTW27EEIM1uScL4TokCIZWwa12VwhxG4hxL2Z0p5CiA+EEJuFEP8EbTPdhkKIXlr5ZUKIXnrnskHO/kKIfzVZvhFC1NC2NxZCHAhq23eCjjlW6zNF2m+xdBmtKHKavtZ264Qoco4OknGVP11NOtsTACllTv0D8oDlQFMgH5gHtE6jPPWBDtrnqsBSoDXwNPCATvnWmswFQBPtt+SlSNZVQK2wba8AfbXPfYGXtc/nAD/iS8XRGZiWpmu9ETg8U9oTOBXoAPyTaBsCBwMrtL81tc81UyDn2YBT+/xykJyNg8uF1TMdOEH7DT8CPVIgp6lrnQqdoCdn2P4BwJPpbk8pZU5a+p2AIinlCillKfAZvlW70oKMvsJYNC4EPpNSlkgpV+JLUNfJfkljypOpK6F1BZZLKWPN2k5pe0opfwe268hgpg27AROklNullDuACUB3u+WUUv4spXRrX//Gl/48Kpqs1aSUU6VPY42k/LfZJmcMol1r23VCLDk1a/1yYFSsOlLRnpCb7h1DK3SlAxG6whjAndqr9Af+V37SK78EfhZCzBJC9NG2JbUSms1cSeiNlGnt6cdsG2aCzDfhszT9NBFCzBFC/CaEOEXb1lCTzU8q5TRzrdPdnqcAm6SUy4K2pa09c1HpG1qhK9WIyBXG3gaaAUcDG/C9/kF65T9JStkB6AHcIYQ4NUbZtLazECIfuAD4QtuUie0Zj2iypbttHwPcwCfapg3AYVLKY4D7gE+FENVIn5xmr3W6+8BVhBonaW3PXFT6GbdCl9BZYUxKuUlK6ZFSeoFhlLsc0ia/lHK99ncz8I0mU6auhNYDmC2l3ASZ2Z5BmG3DtMmsDRqfB1yjuRjQ3CXbtM+z8PnHW2hyBruAUiJnAtc6ne3pBHoCo/3b0t2euaj0ZwDNhRBNNGvwSnyrdqUFzZ8XscJYmP/7YsA/6j8GuFIIUSCEaAI0xze4Y7ecVYQQVf2f8Q3q/UPmroQWYj1lWnuGYbYNxwNnCyFqaq6Ls7VttiKE6A48DFwgpdwftL22ECJP+9wUXxuu0GTdI4TorPXz64N+m51ymr3W6dQJZwL/SikDbpu0t6fVI8OZ8A9fVMRSfE/Qx9Isy8n4XtHmA3O1f+cAHwELtO1jgPpBxzymyb4EG0bvo8jZFF9Uwzxgob/dgEOAicAy7e/B2nYBvKXJuQDomMI2rQxsA6oHbcuI9sT3INoAlOGz3Hon0ob4fOpF2r8bUyRnET7ft7+fvqOVvUTrE/OA2cD5QfV0xKd0lwNvos3yt1lO09fabp2gJ6e2/UPg1rCyaWtPKaVKw6BQKBQViVx07ygUCoUiCkrpKxQKRQVCKX2FQqGoQCilr1AoFBUIpfQVCoWiAqGUvkKhUFQglNJXKBSKCsT/A7VnQ5ejhjwcAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(y_train)\n",
    "plt.plot(y_test,predictions)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lr = LinearRegression()\n",
    "lr.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "metadata": {},
   "outputs": [],
   "source": [
    "rr = Ridge(alpha=0.01)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Ridge(alpha=0.01, copy_X=True, fit_intercept=True, max_iter=None,\n",
       "      normalize=False, random_state=None, solver='auto', tol=0.001)"
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rr.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Ridge(alpha=100, copy_X=True, fit_intercept=True, max_iter=None,\n",
       "      normalize=False, random_state=None, solver='auto', tol=0.001)"
      ]
     },
     "execution_count": 69,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "rr100 = Ridge(alpha=100) #  comparison with alpha value\n",
    "rr100.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_score=lr.score(X_train, y_train)\n",
    "test_score=lr.score(X_test, y_test)\n",
    "Ridge_train_score = rr.score(X_train,y_train)\n",
    "Ridge_test_score = rr.score(X_test, y_test)\n",
    "Ridge_train_score100 = rr100.score(X_train,y_train)\n",
    "Ridge_test_score100 = rr100.score(X_test, y_test)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "linear regression train score: 0.9942440595479894\n",
      "linear regression test score: 0.9944449329037267\n",
      "ridge regression train score low alpha: 0.9942440595180948\n",
      "ridge regression test score low alpha: 0.9944449079888518\n",
      "ridge regression train score high alpha: 0.9915572847411784\n",
      "ridge regression test score high alpha: 0.9915150669097094\n"
     ]
    }
   ],
   "source": [
    "print (\"linear regression train score:\",train_score)\n",
    "print (\"linear regression test score:\", test_score)\n",
    "print (\"ridge regression train score low alpha:\", Ridge_train_score)\n",
    "print (\"ridge regression test score low alpha:\", Ridge_test_score)\n",
    "print (\"ridge regression train score high alpha:\", Ridge_train_score100)\n",
    "print (\"ridge regression test score high alpha:\", Ridge_test_score100)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.legend.Legend at 0x1ee4239f438>"
      ]
     },
     "execution_count": 82,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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+fDDD7n77rvDG0gVM2PGDLp27UpsbCyRkaXvAcrPz+euu+4iMTGRmJgY+vfvT3Z2tuf6qs5rovgfznxP/lyF87yFMSYMjhyB556Db75xfh4JNIdziIwbN459+/axc+dOhg0bxuDBg0/6+ZoSEhK4+eabmTx5st/6CRMmMGfOHJYsWVK0ROyQIUM811d1XhPFw0AfEfkYZ0BbgYtE5DXg98BfwxSfMSe92bMhPR1at3Z+zp5dMceNjIzkxhtvJC8vj++++66oPDU1lUcecSZM2Lp1K5dffjlxcXG0atWKefPmFdvH1q1b6dOnT1H9Sy+9VKpr6sCBA9x55500b96cunXrcumllxZLTBMmTChzCdHXX3+ddu3aERsby6WXXkpOTg4tW7Zk3brQzCOalpbGoEGDipZ0LWnKlCncc889tGjRgri4OCZOnMi8efOKzrOs+qrOU6JQ1c+AK3AGs1/GGcyegLN+xBWquiTI5saYY/Ttt/DBB1C4Zk/Tps7nb78N/7GPHDnCc889BziLB/nzhz/8gYiICDIzM/n888959dVXS9XXqlWLjRs3snjxYqZPL71I5Q033MDq1av5+uuv2bp1K+eddx69e/fm6NGjgLNa3MqVKwPG+fzzzzN27FhmzJjBpk2bWLt2LQMGDKBXr15Fa3kXuvnmm4mPjw/4mjBhQnl+RQDk5OSQmZlJp06/TnzdsmVLYmNjWblyZZn1JwJPD9wBqOpcYK6InAE0AHa6a1YYY8Lk7bchLg4iIpzPERHO57ffho4dw3PMv/71rzz++OPs3buXmjVrMnXqVL9/0W/atIlFixaxdu1a4uLiiIuL44EHHuCSS5xlY7Kysli0aBHr1q0jNjaW2NhYxo0bx2effVa0j+zsbN566y02bNjAaaedBsADDzzA5MmTWbJkCV26dAkaa15eHmPHjmXq1Km0adMGgPbt2zN//nzeeOONUu2fffZZnn322WP+3fiTm+ssqRMXF1esPD4+ntzc3DLrTwSeH7grpKprVfW/liSMCb+rroKcHMjPdz7n5zufr7oqfMe877772LNnD9nZ2Vx22WUsWrTIb7vCvvZmzZoVlTVv/uskDps2OVPD+a466dsWID09HYB27doV/VVft25djh49ysaNG8uMdfHixRw8eJBevXoVleXl5XH77beTmJhY5vahEBMTAzhXFr727NlD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ISGBTp068fPPzurRCxYsoGPHjsTGxlK/fn0uuuiiYuftu7jTypUr6dGjBwkJCbRo0YJHHnmEfHdd3IyMDESE6dOn06ZNG2JiYrjkkkvYsmVLmb+r8ipzridV/VxVL8O5s6mh+4pV1V6q+kXIIzLGAJCxJ4OdB3dS/9T6xcrrn1qfnQd3kr4n4CNOFeLVV19l9uzZZGdn07RpU/7yl78U1d1///3MmTOHefPmsXPnTq677jrS0tLYvdu5kTIqKop//OMf7Nixg++//57NmzczcuTIYvt/+eWXufXWW8nJyeHWW28tM56tW7cyc+ZMzjrrrKKy7du3061bN/r168fmzZv56quvmD9/Pn//u7PWWk5ODpdddhkDBw5k165dPPPMM7zwwgul9v3iiy8yadIk9u3bR9++fZk/fz7XX389kydPZteuXbz22mvccsstfP755wCMGTOGpKQktm3bRnZ2Nq+88grx8fEADB06tOi8Nm3axH333ef3fHJycrj44ovp3r07W7duZe7cubz88stMmlT8EbSZM2fy+eefs2nTJvbv38/9999f5u+qvDyvma2qBaq63X3ZswzGhFnu4VzE/93kiAi5h3P91lWUu+66i6SkJKKiohg2bFjRX/KqyjPPPMNjjz1GixYtiIiI4Prrr6dRo0bMnTsXgC5dutC5c2ciIyNp2LAhd999NwsXLiy2/yuvvJIePXoUjT8EcvbZZxMTE0OjRo3YvXs3b775ZlHdtGnTaN++PX/605+oVasWjRs3ZvTo0UWr9f373/+mTp063HnnndSsWZOOHTty3XXXlTrGjTfeSMeOHRERoqOjeeqppxg5ciRdu3alRo0anHvuufzxj38s2m+tWrXYunUr69evJyIignbt2nHaaacV1a1bt45t27YRFRVF9+7d/Z7X3LlzqVWrFmPHjiUqKorWrVtzzz33MHXq1GLtHnjgAerXr09sbCyDBw8udUUVCp4ThTGmYsVGxaIB1gRTVWKjYis4ouIaNWpU9L527drs3bsXgOzsbPbt20efPn2Ij48veq1fv56sLGfihWXLlpGWlkbDhg2JjY1l0KBB7Nixo9j+va4n/eOPP7J3716++eYbdu3axfr164vq0tPT+fLLL4vFcd1117F161YANm3aRFJSUrGFlJo1K/1UQMlY0tPTefTRR4vt99VXX2XzZmeGoccee4zmzZvTp08fGjVqxF/+8peiNcDnzJnDmjVraNu2LW3atGHy5Ml+z2vjxo0kJycXi61ly5al1gMP9N8hlCxRGFNFNY9vTr3oemQfKP6YUvaBbOpF16N5fNW8s6d+/frUrl2bBQsWsGfPnqLX/v37uffeewEYOHAg55xzDr/88gu5ubm89dZbpfZTo0b5vp5SUlJ45JFHuPHGGzlw4ADgfOlfdNFFxeLIyckp+tJu3LgxmZmZ+C4JnZmZWWYszZo1Y/z48cX2u3fvXj744APAGS95+umnWbt2LV9++SWffvpp0W2v7du3Z+bMmWzfvp0XXniB0aNHs2jRolLHbNq0KRs2bCgW2/r16ytlSVhLFMZUURE1IhjafihREVFk7MlgQ84GMvZkEBURxdD2Q8Nyi2xeXh6HDh0q9iovEWHkyJHceeedrFnjrBiwb98+Pvroo6K/uHNzc4mLiyMmJobMzEwmTJgQkviHDh1K7dq1efrpp4s+L126lJdffplDhw5RUFDA+vXrmTdvHgC9e/dm7969TJo0iaNHj7JixQpeeeWVMo8zatQoJk+ezBdffEF+fj5Hjhxh2bJlRd0+M2fOJD09HVUlLi6OWrVqERkZyZEjR3jttdfIzs5GREhISKBGjRpERpa+AbVXr14cOnSIv/3tbxw5coSff/6ZRx99lOuvvz4kv6vysERhTBWWWDuRkeeP5LqO19G/dX+u63gdo84fFbZbYx988EGio6OLvQq7acq7n759+9K3b19iY2M588wzef755ykoKABgypQpTJ06lZiYGPr168eAAQNCEn9ERATjxo3j0UcfZffu3TRs2JBPPvmE999/n+TkZBISEvj9739f1D0VHx/P3LlzeeONN0hISGDEiBEMGzaMqKiooMe55JJLmDJlCnfddRf169enUaNG3HbbbUVXKt9++y3dunWjTp06nH322ZxzzjnceeedgJNEfvOb31CnTh0uv/xyHnzwQS688MJSx4iLi+Pjjz9mwYIFnHbaaaSlpTF06FBuv/32kPyuykN8L2uqg5SUFA3HYI4xx2rVqlW0bt2ao0dh506oVw9OstvwTyijR49m2bJlfPyxlzlRTzyF/x5LEpFlqup3gle7ojCmAhQUwI4dsH+/89P9w9pUAfPnz2fLli0UFBTwxRdfMGXKFAYN8rzEzknB81Koxphjt3s3HD4M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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.plot(rr.coef_,alpha=0.7,linestyle='none',marker='*',\n",
    "         markersize=5,color='red',label=r'Ridge; $\\alpha = 0.01$',zorder=7) # zorder for ordering the markers\n",
    "plt.plot(rr100.coef_,alpha=0.5,linestyle='none',marker='d',\n",
    "         markersize=6,color='blue',label=r'Ridge; $\\alpha = 100$') # alpha here is for transparency\n",
    "plt.plot(lr.coef_,alpha=0.4,linestyle='none',marker='o',markersize=7,color='green',label='Linear Regression')\n",
    "plt.xlabel('Coefficient Index',fontsize=16)\n",
    "plt.ylabel('Coefficient Magnitude',fontsize=16)\n",
    "plt.legend(fontsize=13,loc=4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [],
   "source": [
    "lasso = Lasso()\n",
    "lasso.fit(X_train,y_train)\n",
    "train_score=lasso.score(X_train,y_train)\n",
    "test_score=lasso.score(X_test,y_test)\n",
    "coeff_used = np.sum(lasso.coef_!=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "training score: 0.9942318835701667\n",
      "test score:  0.9944168429463484\n",
      "number of features used:  1\n"
     ]
    }
   ],
   "source": [
    "print (\"training score:\", train_score) \n",
    "print (\"test score: \", test_score)\n",
    "print (\"number of features used: \", coeff_used)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Lasso(alpha=0.01, copy_X=True, fit_intercept=True, max_iter=1000000.0,\n",
       "      normalize=False, positive=False, precompute=False, random_state=None,\n",
       "      selection='cyclic', tol=0.0001, warm_start=False)"
      ]
     },
     "execution_count": 89,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lasso001 = Lasso(alpha=0.01, max_iter=10e5)\n",
    "lasso001.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "metadata": {},
   "outputs": [],
   "source": [
    "train_score001=lasso001.score(X_train,y_train)\n",
    "test_score001=lasso001.score(X_test,y_test)\n",
    "coeff_used001 = np.sum(lasso001.coef_!=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "training score for alpha=0.01: 0.9942440583303916\n",
      "test score for alpha =0.01:  0.9944447728683314\n",
      "number of features used: for alpha =0.01: 1\n"
     ]
    }
   ],
   "source": [
    "print (\"training score for alpha=0.01:\", train_score001)\n",
    "print (\"test score for alpha =0.01: \", test_score001)\n",
    "print (\"number of features used: for alpha =0.01:\", coeff_used001)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "metadata": {},
   "outputs": [],
   "source": [
    "lasso00001 = Lasso(alpha=0.0001, max_iter=10e5)\n",
    "lasso00001.fit(X_train,y_train)\n",
    "train_score00001=lasso00001.score(X_train,y_train)\n",
    "test_score00001=lasso00001.score(X_test,y_test)\n",
    "coeff_used00001 = np.sum(lasso00001.coef_!=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "training score for alpha=0.0001: 0.9942440595478677\n",
      "test score for alpha =0.0001:  0.9944449313154592\n",
      "number of features used: for alpha =0.0001: 1\n"
     ]
    }
   ],
   "source": [
    "print (\"training score for alpha=0.0001:\", train_score00001 )\n",
    "print (\"test score for alpha =0.0001: \", test_score00001)\n",
    "print (\"number of features used: for alpha =0.0001:\", coeff_used00001)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "metadata": {},
   "outputs": [],
   "source": [
    "lr = LinearRegression()\n",
    "lr.fit(X_train,y_train)\n",
    "lr_train_score=lr.score(X_train,y_train)\n",
    "lr_test_score=lr.score(X_test,y_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "LR training score: 0.9942440595479894\n",
      "LR test score:  0.9944449329037267\n"
     ]
    }
   ],
   "source": [
    "print (\"LR training score:\", lr_train_score)\n",
    "print (\"LR test score: \", lr_test_score)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.legend.Legend at 0x1ee42e877f0>"
      ]
     },
     "execution_count": 96,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.subplot(1,2,1)\n",
    "plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\\alpha = 1$',zorder=7) # alpha here is for transparency\n",
    "plt.plot(lasso001.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Lasso; $\\alpha = 0.01$') # alpha here is for transparency\n",
    "\n",
    "plt.xlabel('Coefficient Index',fontsize=16)\n",
    "plt.ylabel('Coefficient Magnitude',fontsize=16)\n",
    "plt.legend(fontsize=13,loc=4)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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MVx9tftMIE8++SSnnHIKqampZGZmMnLkSIqKDq138NRTT9G6dWtSU1M54YQT+P3vfx82HmDnzp0MGjSIZs2a0bRpUwYPHsyuXYemuI8ZM4b27dtXqO2f//wn7du3Jy0tjYsvvpi9e/fStm1bVq9eHfV59+rVi2uvvZY2bdpEXdcxhaomXVi4cOFaVZ0bLixdulRD0ru3art2qtnZ7rN379B5Y0C3bt30T3/6U9C0yZMna25urhYXF+v8+fO1cePGOmHCBFVVXbFihdauXVuXLFmiqqq7d+/W2bNnh4wvoVevXtqnTx/dtWuX7tq1S3NycjQnJycizc8884y2atVKly5dqnl5edq2bVu96KKL9I477jgs76233qr16tULGR599NGQx/n444+1WrVqEWk7mli2bFnINGCuRvjb95cJTsItSRMs7XzgxEgPHM8QtUEvWqTaoYNqnz7uc/Hi0HljQDiDLs/w4cP1yiuvVFXV1atX63HHHadvvPGG5ufnl+YJFa+qumnTJgX0+++/L4377rvvFNDNmzf70nDw4EFt0KCB/vvf/y6N69evn6ampuq2bdt81eEXM+jIfvuRuPG9NERaH+DxaFoJRxzt20NKCmza5D7POCNhUl577TU6d+5MgwYNqFevHn/729/Yvn07AG3atOGVV17h73//O82bN+fcc89l2rRpIeMBNmzYAEDr1od8VrRt27ZMWkV8/vnn7N+/n969D/llLCws5J577qFRo0YxOW+jcvg16E7ApyHSPsX5604uevaEBx5wnwliw4YN3HDDDTzwwANs2bKFvXv3cvvtt5e0jADo168f06dPZ8eOHVx11VX07duXn376KWR8y5YtAVi7dm1pHbm5uQClaX50NW3alBo1agCwcuVKPvjgA84I8cf3m9/8hrp164YMo0ePrszlMYLg16BTcf6yg3EQN0EjuRgzBvr1c59VQGFhIQUFBWXCvn37KC4uplGjRtSoUYOvvvqKl19+ubTMihUr+OCDD/jpp5+oUaMG9erVQ0RKDax8fEpKCs2bN6dnz54MHz6cPXv2sHv3boYPH84ll1xCs2bNAHjooYfIysoKqbVly5asW7eOBQsWsGvXLgYOHEj9+vVD3uEnTJjAvn37QobADrsSioqKKCgo4MABN1O35JoE/pkZQfDTLgeWAmNDpI0Fvou0rR/PEPUzdBXT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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.subplot(1,2,2)\n",
    "plt.plot(lasso.coef_,alpha=0.7,linestyle='none',marker='*',markersize=5,color='red',label=r'Lasso; $\\alpha = 1$',zorder=7) # alpha here is for transparency\n",
    "plt.plot(lasso001.coef_,alpha=0.5,linestyle='none',marker='d',markersize=6,color='blue',label=r'Lasso; $\\alpha = 0.01$') # alpha here is for transparency\n",
    "plt.plot(lasso00001.coef_,alpha=0.8,linestyle='none',marker='v',markersize=6,color='black',label=r'Lasso; $\\alpha = 0.00001$') # alpha here is for transparency\n",
    "plt.plot(lr.coef_,alpha=0.7,linestyle='none',marker='o',markersize=5,color='green',label='Linear Regression',zorder=2)\n",
    "plt.xlabel('Coefficient Index',fontsize=16)\n",
    "plt.ylabel('Coefficient Magnitude',fontsize=16)\n",
    "plt.legend(fontsize=13,loc=4)\n",
    "plt.tight_layout()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3.7.3",
   "language": "python",
   "name": "py37"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
