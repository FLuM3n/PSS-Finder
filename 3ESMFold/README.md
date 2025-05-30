we uesd ESMFold API to predict the structure of certain protein sequence to make the framework be simplier

but the length and the stability is not garanteened, so you can download the ESMFold for local deployment.  

[ESMFold](https://github.com/facebookresearch/esm) 

Citations
If you find the models useful in your research, we ask that you cite the relevant paper:

@article{rives2019biological,
  author={Rives, Alexander and Meier, Joshua and Sercu, Tom and Goyal, Siddharth and Lin, Zeming and Liu, Jason and Guo, Demi and Ott, Myle and Zitnick, C. Lawrence and Ma, Jerry and Fergus, Rob},
  title={Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences},
  year={2019},
  doi={10.1101/622803},
  url={https://www.biorxiv.org/content/10.1101/622803v4},
  journal={PNAS}
}
For the self-attention contact prediction:

@article{rao2020transformer,
  author = {Rao, Roshan M and Meier, Joshua and Sercu, Tom and Ovchinnikov, Sergey and Rives, Alexander},
  title={Transformer protein language models are unsupervised structure learners},
  year={2020},
  doi={10.1101/2020.12.15.422761},
  url={https://www.biorxiv.org/content/10.1101/2020.12.15.422761v1},
  journal={bioRxiv}
}
For the MSA Transformer:

@article{rao2021msa,
  author = {Rao, Roshan and Liu, Jason and Verkuil, Robert and Meier, Joshua and Canny, John F. and Abbeel, Pieter and Sercu, Tom and Rives, Alexander},
  title={MSA Transformer},
  year={2021},
  doi={10.1101/2021.02.12.430858},
  url={https://www.biorxiv.org/content/10.1101/2021.02.12.430858v1},
  journal={bioRxiv}
}
For variant prediction using ESM-1v:

@article{meier2021language,
  author = {Meier, Joshua and Rao, Roshan and Verkuil, Robert and Liu, Jason and Sercu, Tom and Rives, Alexander},
  title = {Language models enable zero-shot prediction of the effects of mutations on protein function},
  year={2021},
  doi={10.1101/2021.07.09.450648},
  url={https://www.biorxiv.org/content/10.1101/2021.07.09.450648v1},
  journal={bioRxiv}
}
For inverse folding using ESM-IF1:

@article{hsu2022learning,
	author = {Hsu, Chloe and Verkuil, Robert and Liu, Jason and Lin, Zeming and Hie, Brian and Sercu, Tom and Lerer, Adam and Rives, Alexander},
	title = {Learning inverse folding from millions of predicted structures},
	year = {2022},
	doi = {10.1101/2022.04.10.487779},
	url = {https://www.biorxiv.org/content/early/2022/04/10/2022.04.10.487779},
	journal = {ICML}
}
For the ESM-2 language model and ESMFold:

@article{lin2022language,
  title={Language models of protein sequences at the scale of evolution enable accurate structure prediction},
  author={Lin, Zeming and Akin, Halil and Rao, Roshan and Hie, Brian and Zhu, Zhongkai and Lu, Wenting and Smetanin, Nikita and dos Santos Costa, Allan and Fazel-Zarandi, Maryam and Sercu, Tom and Candido, Sal and others},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory}
}
Much of this code builds on the fairseq sequence modeling framework. We use fairseq internally for our protein language modeling research. We highly recommend trying it out if you'd like to pre-train protein language models from scratch.

Additionally, if you would like to use the variant prediction benchmark from Meier et al. (2021), we provide a bibtex file with citations for all data in ./examples/variant-prediction/mutation_data.bib. You can cite each paper individually, or add all citations in bulk using the LaTeX command:

\nocite{wrenbeck2017deep,klesmith2015comprehensive,haddox2018mapping,romero2015dissecting,firnberg2014comprehensive,deng2012deep,stiffler2015evolvability,jacquier2013capturing,findlay2018comprehensive,mclaughlin2012spatial,kitzman2015massively,doud2016accurate,pokusaeva2019experimental,mishra2016systematic,kelsic2016rna,melnikov2014comprehensive,brenan2016phenotypic,rockah2015systematic,wu2015functional,aakre2015evolving,qi2014quantitative,matreyek2018multiplex,bandaru2017deconstruction,roscoe2013analyses,roscoe2014systematic,mavor2016determination,chan2017correlation,melamed2013deep,starita2013activity,araya2012fundamental}
License
This source code is licensed under the MIT license found in the LICENSE file in the root directory of this source tree.

ESM Metagenomic Atlas (also referred to as “ESM Metagenomic Structure Atlas” or “ESM Atlas”) data is available under a CC BY 4.0 license for academic and commercial use. Copyright (c) Meta Platforms, Inc. All Rights Reserved. Use of the ESM Metagenomic Atlas data is subject to the Meta Open Source Terms of Use and Privacy Policy.

