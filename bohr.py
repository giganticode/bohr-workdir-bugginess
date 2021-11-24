from bohrapi.artifacts import Commit
from bohrlabels.labels import CommitLabel
from bohrapi.core import Dataset, Task, Workspace


HEURISTICS_CLASSIFIER = 'bugginess/keywords/bug_keywords_lookup_in_message.py:bugginess/keywords/buggless_keywords_lookup_in_message.py' # run all heuristics
BOHR_FRAMEWORK_VERSION = '0.5.0rc0'

commits_200k = Dataset(id='bohr.200k_commits', top_artifact=Commit,)

berger = Dataset(id='manual_labels.berger',
                 top_artifact=Commit,
                 get_bohr_label_for_datapoint=lambda c: (CommitLabel.BugFix if c.raw_data['manual_labels']['berger']['bug'] == 1 else CommitLabel.NonBugFix))

levin = Dataset(id='manual_labels.levin',
                 top_artifact=Commit,
                 get_bohr_label_for_datapoint=lambda c: (CommitLabel.BugFix if c.raw_data['manual_labels']['levin']['bug'] == 1 else CommitLabel.NonBugFix))

herzig = Dataset(id='manual_labels.herzig',
                 top_artifact=Commit,
                 get_bohr_label_for_datapoint=lambda c: (CommitLabel.BugFix if c.raw_data['manual_labels']['herzig']['bug'] == 1 else CommitLabel.NonBugFix))

bugginess = Task(name='bugginess', author='hlib', description='bug or not', top_artifact=Commit,
                 labels=[CommitLabel.NonBugFix, CommitLabel.BugFix],
                 training_dataset=commits_200k,
                 test_datasets=[levin, berger, herzig], heuristics_classifier=f'{HEURISTICS_CLASSIFIER}@29dfeed9fe2a9f43b6dd8583b141696c4db178d5')

w = Workspace(BOHR_FRAMEWORK_VERSION, [bugginess])

