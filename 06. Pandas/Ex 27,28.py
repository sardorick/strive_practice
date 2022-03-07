# The next three puzzles are slightly harder...

# **26.** You have a DataFrame that consists of 10 columns of floating--point numbers. Suppose that exactly 5 entries in each row are NaN values. For each row of the DataFrame, find the *column* which contains the *third* NaN value.

# (You should return a Series of column labels. Try not to use `.iterrows()`)
data = {
1:[np.nan,2,np.nan,4,5],
2:[np.nan,2,3,np.nan,5],
3:[np.nan,2,np.nan,4,5],
4:[1,np.nan,3,np.nan,np.nan],
5:[1,2,np.nan,4,5],
6:[1,np.nan,3,np.nan,np.nan],
7:[1,2,3,4,5],
8:[1,np.nan,3,np.nan,np.nan],
9:[1,2,3,4,np.nan],
10:[1,np.nan,3,4,5]}

labels = ['a', 'b', 'c', 'd','e']
df = pd.DataFrame(data, index=labels)
df 

# **27.** A DataFrame has a column of groups 'grps' and and column of numbers 'vals'. For example: 

# ```python
# df = pd.DataFrame({'grps': list('aaabbcaabcccbbc'), 
#                    'vals': [12,345,3,1,45,14,4,52,54,23,235,21,57,3,87]})
# ```
# For each *group*, find the sum of the three greatest values.

df = pd.DataFrame({'grps': list('aaabbcaabcccbbc'), 
                   'vals': [12,345,3,1,45,14,4,52,54,23,235,21,57,3,87]})

df2 = df.groupby(['grps']).vals.nlargest(3).sum(level=0)
df2
