import pandas as pd

raw = pd.read_csv('compas-scores-raw.csv')
new = raw[['Person_ID','AssessmentID','Case_ID','DecileScore','ScoreText','Scale_ID','DisplayText','FirstName','LastName','DateOfBirth','Sex_Code_Text','RecSupervisionLevel','RecSupervisionLevelText']]
#new = raw.drop(columns = ['MiddleName','IsCompleted','IsDeleted','AssessmentReason'])
#drop rows with nan in column 'ScoreText'
new = new.dropna(subset=['ScoreText'])
new = new.rename(columns={'Sex_Code_Text':'Sex'})
new['Sex'] = new['Sex'].replace({"Female": '2', "Male": '1'})
#add id column
new = new.reset_index(drop=True)
new['ID'] = new.index
#delet space for each cell
new = new.applymap(lambda x: x.replace(' ', '') if type(x) == str else x)

print(new.head(2))
new.to_csv('compas-scores-prepocess.csv',index=False)