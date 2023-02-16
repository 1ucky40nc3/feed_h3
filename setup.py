from setuptools import find_packages, setup

setup(
    name="feed_h3",
    version="0.0.1",
    author="Louis Wendler",
    description="Feed Hungry Hungry Hippos (H3) - Do Languange Modeling with a 🦛 (unofficial)",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="python nlp machine-learning deep-learning jupyter-notebook language-modeling pytorch accelerate datasets state-space-model",
    license="MIT License",
    url="https://github.com/1ucky40nc3/feed_h3",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
)

