# Art of Text

Neural style transfer for written text.

## Getting Started

To train a model, first create a `tmp/` folder in `language-style-transfer`, then go to the `code/` folder and run the following command:
```bash
python style_transfer.py --train ../data/yelp/sentiment.train --dev ../data/yelp/sentiment.dev --output ../tmp/sentiment.dev --vocab ../tmp/yelp.vocab --model ../tmp/model
```

To test the model, run the following command:
```bash
python style_transfer.py --test ../data/yelp/sentiment.test --output ../tmp/sentiment.test --vocab ../tmp/yelp.vocab --model ../tmp/model --load_model true
```

The model and results will be saved in the `tmp/` folder.

Check `code/options.py` for all running options.

To train a HPL -> Shakespeare converter create a `style-tmp` here in root:

```bash
python style_transfer.py --train ../../data/preProcessCode/horror_shake.train --dev ../../data/preProcessCode/horror_shake.dev --output ../../style-tmp/style.dev --vocab ../../style-tmp/style.vocab --model ../../style-tmp/model
```

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you have to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Built With

* [Tensorflow](https://www.tensorflow.org/) - Deep learning framework


## Authors

* **Yuriy Minin** - [University of Texas at Austin](https://www.utexas.edu/)
* **Mukundan Kuthalam** - [University of Texas at Austin](https://www.utexas.edu/)
* **Zohaib Imam** - [University of Texas at Austin](https://www.utexas.edu/)
* **Ram Muthukumar** - [University of Texas at Austin](https://www.utexas.edu/)

## License

This project is licensed under the GPLv3 License - see the [LICENSE.txt](LICENSE.txt) file for details
