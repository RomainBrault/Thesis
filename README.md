DATA ARE NOT REAL! [![Build Status](https://travis-ci.com/RomainBrault/Thesis.svg?token=BGkmfYrnrsiGdq17pxis&branch=master)](https://travis-ci.com/RomainBrault/Thesis)
==============

# About

My Ph.D. thesis manuscript at Université d'Évry val d'Essonne and Télécom-ParisTech on operator-valued kernel approximation, supervised by [Florence d'Alché-Buc](http://perso.telecom-paristech.fr/~fdalche/Site/index.html).

## Français

Manuscript de thèse de doctorat de l'université d'Évry val d'Essonne et Télécom-ParisTech sur l'approximation de noyaux à valeurs opérateur. Thèse encadrée par [Florence d'Alché-Buc](http://perso.telecom-paristech.fr/~fdalche/Site/index.html).

# Download

To download the latest version of the thesis manuscript click [here](https://github.com/RomainBrault/Thesis/raw/master/ThesisRomainBrault.pdf). This document is digitally signed using [pgp](https://fr.wikipedia.org/wiki/Pretty_Good_Privacy). To obtain the public key run

    gpg --keyserver pgp.mit.edu --recv-keys A276D73294A106E2544FFF9E3E5B5D0B181C5E04

To check the document run

    gpg --verify ThesisRomainBrault.pdf.asc ThesisRomainBrault.pdf

# Abstract

In this thesis we study scalable methods to perform regression with Operator-Valued Kernels (OVKs) in order to learn vector-valued functions.

When data present structure, or relations between them or their different components, a common approach is to treat the data as a vector living in an appropriate vector space rather a collection of real number. This representation allows to take into account the structure of the data by defining an appropriate space embbeding the underlying structure. Thus many problems in machine learning can be cast into learning vector-valued functions. Operator-Valued Kernels OVKs and vector-valued Reproducing Kernel Hilbert Spaces provide a theoretical and practical framework to address that issue, naturally extending the well-known framework of scalar-valued kernels. In the context of scalar-valued function learning, a scalar-valued kernel can be seen a a similarity measure between two data point. A solution of the learning problem has the form of a linear combination of theses similarities with respect to weights to determine in order to have the best "fit" of the data. When dealing with OVKs, the evalution of the kernel is no longer a scalar similarity, but a function acting on vectors. A solution is then a linear combination of operators with respect to vector weights.

Although OVKs generalize strictly scalar-valued kernels, large scale applications are usually not affordable with these tools that require an important computational power along with a large memory capacity. In this thesis, we propose and study scalable methods to perform regression with OVKs. To achieve this goal, we extend Random Fourier Features, an approximation technique originally introduced for scalar-valued kernels, to OVKs. The idea is to take advantage of an approximated operator-valued feature map in order to come up with a linear model in a finite dimensional space.

First we develop a general framework devoted to the approximation of shift-invariant Mercer kernels on Locally Compact Abelian groups and study their properties along with the complexity of the algorithms based on them. Second we show theoretical guarantees by bounding the error due to the approximation, with high probability. Third, we study various applications of Operator Random Fourier Features to different tasks of Machine learning such as multi-class classification, multi-task learning, time serie modeling, functional regression and anomaly detection. We also compare the proposed framework with other state of the art methods. Fourth, we conclude by drawing short-term and mid-term perspectives.

# Compile from sources

To pull the latest version and compile the thesis locally run `./compile -f`.
To synchronize Overleaf, Git and push back on both run `./update`. This requires writing permission on both Overleaf and Git repository.

# Contact

Université Paris-Saclay ED STIC -- 580, Université Paris Sud, Bâtiment 650 Ada Lovelace, 91405 Orsay Cedex, France.

For any questions/remarks please raise an issue to keep track of it. In case it is not possible for some reasons please contact [Romain Brault](mailto:ro.brault@gmail.com).

# Thanks

This document was typeset using the typographical look-and-feel classicthesis developed by André Miede. The style was inspired by Robert Bringhurst's seminal book on typography "The Elements of Typographic Style". classicthesis is available at [https://bitbucket.org/amiede/classicthesis/](https://bitbucket.org/amiede/classicthesis/) for both LaTeX and Lyx.

# Licence
Copyright (c) <2016> <Romain Brault romain.brault@telecom-paritech.fr,
                      Florence d'Alche-Buc florence.dalche@telecom-paristech.fr,
                      Universite d'Evry val d'Essone, Telecom-ParisTech>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
