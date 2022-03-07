# **23.** Given a DataFrame of numeric values, say
```python
df = pd.DataFrame(np.random.random(size=(5, 3))) # a 5x3 frame of float values
```

how do you subtract the row mean from each element in the row?
df = pd.DataFrame(np.random.random(size=(5, 3))) # a 5x3 frame of float values
df.mean(axis=1)


# **24.** Suppose you have DataFrame with 10 columns of real numbers, for example:

```python
df = pd.DataFrame(np.random.random(size=(5, 10)), columns=list('abcdefghij'))
```
Which column of numbers has the smallest sum? (Find that column's label.)

df = pd.DataFrame(np.random.random(size=(5, 10)), columns=list('abcdefghij'))
df2 =  df[list('abcdefghij')].sum()
df3 = df2.sort_values(axis=0)
df3.head(1)