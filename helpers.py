import pandas as pd
import seaborn as sns

COLOR_MAP = {
	"Allantois" : "#532C8A",
	"Anterior Primitive Streak" : "#c19f70",
	"Blood progenitors 1" : "#f9decf",
	"Blood progenitors 2" : "#c9a997",
	"Cardiomyocytes" : "#B51D8D",
	"Caudal epiblast" : "#9e6762",
	"Caudal Mesoderm" : "#3F84AA",
	"Def. endoderm" : "#F397C0",
	"Nascent mesoderm" : "#C594BF",
	"Mixed mesoderm" : "#DFCDE4",#
	"Endothelium" : "#eda450",
	"Epiblast" : "#635547",
	"Erythroid1" : "#C72228",
	"Erythroid2" : "#EF4E22",
	"Erythroid3" : "#f77b59",
	"ExE ectoderm" : "#989898",
	"ExE endoderm" : "#7F6874",
	"ExE mesoderm" : "#8870ad",
	"Rostral neurectoderm" : "#65A83E",
	"Forebrain/Midbrain/Hindbrain" : "#647a4f",
	"Gut" : "#EF5A9D",
	"Haematoendothelial progenitors" : "#FBBE92",
	"Caudal neurectoderm": "#354E23",
	"Intermediate mesoderm" : "#139992",
	"Neural crest": "#C3C388",
	"NMP" : "#8EC792",
	"Notochord" : "#0F4A9C",
	"Paraxial mesoderm" : "#8DB5CE",
	"Parietal endoderm" : "#1A1A1A",
	"PGC" : "#FACB12",
	"Pharyngeal mesoderm" : "#C9EBFB",
	"Primitive Streak" : "#DABE99",
	"Mesenchyme" : "#ed8f84",
	"Somitic mesoderm" : "#005579",
	"Spinal cord" : "#CDE088",
	"Surface ectoderm" : "#BBDCA8",
	"Visceral endoderm" : "#F6BFCB",
	"Mes1": "#c4a6b2",
	"Mes2":"#ca728c",
	"Cardiomyocytes" : "#B51D8D",
}

def transition_scores(transition_scores_raw, model_names=None, 
                      model_label='Model', raw=False, custom_colors=None, 
                      height=4, aspect=1, ax=None):

    if model_names == None:
        if raw == False:
            print('a')
        else:
            score = []
            transition = []
            embedding=[]
            for key in transition_scores_raw.keys():
                for item in transition_scores_raw[key]:
                    score.append(item)
                    transition.append(key[0] + r'$\rightarrow$' + key[1])
            
            transition_scores_ = pd.DataFrame(
                {'CBDir score':score, ' ':transition})
            
            PROPS = {
                'boxprops':{'facecolor':'darkgrey', 'edgecolor':'black'},
                'medianprops':{'color':'black'},
                'whiskerprops':{'color':'black'},
                'capprops':{'color':'black'}
            }
            
            ax = sns.boxplot(
                data=transition_scores_, 
                y=' ', x="CBDir score", 
                orient='h', fliersize=0, **PROPS, showmeans=True, ax=ax,
                meanprops={
                    "marker":"o",
                    "markerfacecolor":"white", 
                    "markeredgecolor":"black",
                    "markersize":"8"
                })