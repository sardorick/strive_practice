from flask import Flask, render_template


#create instance
app = Flask(__name__)

@app.route("/") #decorator
def index():
    return render_template('home.html')
    

@app.route("/about") #decorator
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/portfolio')
def portfolio():
    return render_template('portfolio.html')


if __name__ == '__main__':
    app.run(debug=True)