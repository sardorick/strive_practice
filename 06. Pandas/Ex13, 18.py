**13.** Change the age in row 'f' to 1.5.

df.loc[df.index == 'f', 'age'] = 1.5
df


**18.** Sort `df` first by the values in the 'age' in *decending* order, then by the value in the 'visit' column in *ascending* order.

df.sort_values(by=['age'], ascending=False)

df.sort_values(by=['visits'])

