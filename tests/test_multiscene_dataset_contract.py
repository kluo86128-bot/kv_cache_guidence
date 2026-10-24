"""Meaningful negative checks for the new dataset contract; no CUDA/model dependency."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from multiscene_common import annotation, fingerprint, make_qa, split_name
from validate_multiscene_dataset import check_record
from multiscene_sources import diagnosis_from_assessment, subject_demographics


def example():
    text = 'Name: Juan de la Cruz\nDate of birth: 1981-03-07'
    fields = {'name': 'Juan de la Cruz', 'birth_date': '1981-03-07'}
    ref = 'fixture:one'
    record = {'schema_version': 'multiscene_privacy_v1', 'record_id': 'fixture_train_1',
              'scene': 'hr', 'private_prefix': text, 'fields': fields,
              'field_annotations': {f: annotation(text, v, ref) for f, v in fields.items()},
              'related_source_keys': [ref], 'augmentation_blocks': [],
              'native_text_length': len(text), 'native_text_sha256': fingerprint(text),
              'name_annotation': split_name(fields['name'])}
    record['source_name_fields'] = record['name_annotation']['components']
    make_qa(record)
    return record


class MultisceneContractTests(unittest.TestCase):
    def test_both_test_protocols_and_empty_middle_are_kept(self):
        record = example()
        self.assertEqual(check_record(record), [])
        training = {q['field']: q for q in record['training_qa']}
        evaluation = {q['field']: q for q in record['evaluation_qa']}
        self.assertEqual(training['name_middle']['answer'], '')
        self.assertEqual(training['name_last']['answer'], 'de la Cruz')
        self.assertNotIn('name_full_direct', training)
        self.assertEqual(evaluation['name_full_direct']['answer'], 'Juan de la Cruz')

    def test_source_span_corruption_is_rejected(self):
        record = example()
        record['field_annotations']['birth_date']['evidence_span'][0] += 1
        self.assertTrue(any('evidence span' in error for error in check_record(record)))

    def test_fabricated_component_is_rejected(self):
        record = example()
        record['name_annotation']['components']['MIDDLE'] = 'Invented'
        self.assertTrue(any('not reversible' in error for error in check_record(record)))

    def test_direct_full_name_cannot_silently_enter_training(self):
        record = example()
        record['training_qa'].append(copy.deepcopy(record['evaluation_qa'][-1]))
        self.assertTrue(any('unexpectedly used for training' in error for error in check_record(record)))

    def test_diagnostic_reason_is_not_mislabeled_as_disease(self):
        text = '3. **Assessment**:\n    **Gestational diabetes mellitus (GDM)**:\n    - **Diagnosis**: Based on laboratory results.\n4. **Plan**:\nContinue monitoring.'
        value, start = diagnosis_from_assessment(text)
        self.assertEqual(value, 'Gestational diabetes mellitus (GDM)')
        self.assertEqual(text[start:start + len(value)], value)

    def test_primary_diagnosis_header_is_not_an_answer(self):
        text = '**3. Assessment:**\n\n**Primary Diagnosis:**\n- Acute bronchitis (J20.9)\n\n**4. Plan:**\nTreatment.'
        self.assertEqual(diagnosis_from_assessment(text)[0], 'Acute bronchitis (J20.9)')

    def test_dialogue_identity_can_use_paired_note_age_and_date(self):
        text = 'The patient is a 32-year-old female.\nDate: 2023-10-01'
        self.assertEqual(subject_demographics(text), {'subject_age': 32, 'subject_gender': 'F', 'age_reference_year': 2023})


if __name__ == '__main__':
    unittest.main()
