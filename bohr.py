from bohrapi.artifacts import Commit
from bohrlabels.labels import CommitLabel
from bohrapi.core import Dataset, Task, Workspace, Experiment

commits_200k = Dataset(id='bohr.200k_commits', top_artifact=Commit,)

berger = Dataset(id='manual_labels.berger',
                 top_artifact=Commit,
                 get_bohr_label_for_datapoint=lambda c: (CommitLabel.BugFix if c.raw_data['manual_labels']['berger']['bug'] == 1 else CommitLabel.NonBugFix))

levin = Dataset(id='manual_labels.levin',
                 top_artifact=Commit,
                 get_bohr_label_for_datapoint=lambda c: (CommitLabel.BugFix if c.raw_data['manual_labels']['levin']['bug'] == 1 else CommitLabel.NonBugFix))

herzig = Dataset(id='manual_labels.herzig',
                 top_artifact=Commit,
                 get_bohr_label_for_datapoint=lambda c: (CommitLabel.BugFix if c.raw_data['manual_labels']['herzig']['CLASSIFIED'] == 'BUG' else CommitLabel.NonBugFix))

bugginess = Task(name='bugginess', author='hlib', description='bug or not', top_artifact=Commit,
                 labels=[CommitLabel.NonBugFix, CommitLabel.BugFix],
                 training_dataset=commits_200k,
                 test_datasets=[levin, berger, herzig], )

exp = Experiment('only_message_keywords', bugginess, heuristics_classifier=f'bugginess/keywords/bug_keywords_lookup_in_message.py:bugginess/keywords/buggless_keywords_lookup_in_message.py@988e934bcd9447d18ddf4af8957ceef286c8d2d7')

w = Workspace('0.5.0rc0', [exp])

