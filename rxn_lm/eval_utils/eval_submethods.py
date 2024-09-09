"""
As noted in some of the comments below, some of the code is adapted from the Hugging Face Transformer library. This has
the following licence:
Copyright 2018- The Hugging Face team. All rights reserved.

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""


import collections
import functools
import warnings
from dataclasses import dataclass
from typing import Optional, Callable, Union, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from transformers import trainer_pt_utils

from .. import chem_ops


@functools.lru_cache(int(1e7))  # <- effectively get it to memoize
def clean_pred_func(pred, put_in_frozen_set=False):
    """
    Remove white space and then canonicalize.
    """
    pred = pred.strip()

    def canonicalize_ignore_errors(smiles_):
        try:
            smiles_ = chem_ops.canonicalize(smiles_)
        except:
            pass
        return smiles_
    pred = canonicalize_ignore_errors(pred)

    if put_in_frozen_set:
        pred = frozenset((canonicalize_ignore_errors(el[0]), el[1]) for el in
                         collections.Counter(pred.split('.')).items())
        # ^ canonicalize may be redundant but cannot work out how rdkit canonicalizes multiple molecule SMILES so being
        # defensive here. could in future consider moving it before the counter.

    return pred


@dataclass
class PredictionStepArgs:
    teacher_forcing: bool = True

    max_length: Optional[int] = None
    num_beams: Optional[int] = None
    num_return_sequences: Optional[int] = None

    # Collect various other outputs if set:
    manually_compute_per_item_losses: bool = False
    collect_encoder_outputs_type: Optional[str] = None
    collect_input_ids: bool = False
    encoder_embeddings_op: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    # ^ if defined will be run on the encoder embeddings before returning them (e.g., to get the mean of the embeddings)

    def __post_init__(self):
        if (self.num_beams is not None) and self.teacher_forcing:
            raise RuntimeError("Set beam length while using teacher forcing!")

        if self.collect_encoder_outputs_type:
            # None (or empty string) will mean the embeddings are not collected.
            if self.collect_encoder_outputs_type == "all":
                pass
            elif self.collect_encoder_outputs_type == "avg":
                self.encoder_embeddings_op = lambda x: x.mean(dim=1)
            else:
                raise NotImplementedError(f"collect_encoder_outputs_type={self.collect_encoder_outputs_type} "
                                          f"not implemented.")


@dataclass
class PredictionOutput:
    """
    This class holds the collection of prediction outputs at each step.

    Has a group of methods able to merge together multiple prediction outputs (concatenating and padding as
    appropriate). Also can convert tensor representation to numpy representation instead.
    """
    pseudo_per_item_losses: Optional[Union[torch.Tensor, np.ndarray]] = None
    # ^ individual losses for every item in the batch, pseudo as may just be average for the batch.
    generated_tokens: Optional[Union[torch.Tensor, np.ndarray]] = None
    # ^ tokens/predictions from the model. note that may be several items for each label if returning
    # top-k predictions and k>1.
    labels: Optional[Union[torch.Tensor, np.ndarray]] = None

    # Below we have the ones that _might_ be present depending on PredictionStepArgs

    manually_computed_losses_per_item_per_token: Optional[Union[torch.Tensor, np.ndarray]] = None
    # ^ individual losses for every item in the batch for every token, computed from the logits and the labels.
    # which differs from `pseudo_per_item_losses` as that is computed automatically in the Hugging Face model forward pass.
    # manually_computed_losses_per_item_per_token this will only be populated if `manually_compute_per_item_losses` in
    # PredictionStepArgs is True.

    encoder_attn_zerod_outputs: Optional[Union[torch.Tensor, np.ndarray]] = None
    # ^ if "all" then zero'd out where the attention mask says they are zero'd out
    # (i.e., only the ones that the decoder can see).

    input_ids: Optional[Union[torch.Tensor, np.ndarray]] = None

    pad_token_input_id: int = 1

    _is_np: Optional[bool] = None

    HELD_OTHERS_PREFIX = '_held__'

    def repeat_losses(self, expand_size):
        self.pseudo_per_item_losses = self.pseudo_per_item_losses.repeat(expand_size)

    def add_another(self, other):
        """
        Concatenate another `PredictionOutput` dataclass to this one.

        Converts other's Pytorch Tensors to Numpy arrays if `self.is_np` is True
        """
        #depreciated
        for item_name, pad_value in self._property_name_and_pad_value:
            others_item = getattr(other, item_name)
            selfs_item = getattr(self, item_name)
            others_item = self._cpufy_and_numpify_if_appropriate(others_item)
            new_item = others_item if selfs_item is None else self._pad_and_concat(selfs_item, others_item,
                                                                                   pad_value=pad_value)
            setattr(self, item_name, new_item)

    def hold_others_but_do_not_cat_or_pad(self, other):
        """cheaper version where stores others but does not concatenate or pad until `pad_and_concat_held_others` called"""
        for item_name, _ in self._property_name_and_pad_value:
            others_item = getattr(other, item_name)

            held_name = self._get_held_name_from_prop_name(item_name)
            held = getattr(self, held_name, None)
            if held is None:
                held = []
                setattr(self, held_name, held)

            others_item = self._cpufy_and_numpify_if_appropriate(others_item)
            held.append(others_item)

    def pad_and_concat_held_others(self):
        for item_name, pad_value in self._property_name_and_pad_value:

            # Get current held items -- if empty then nothing to do. If not empty add the non held item to the beginning
            # of the list.
            held_name = self._get_held_name_from_prop_name(item_name)
            held = getattr(self, held_name, None)
            if held is None:
                continue
            else:
                current = getattr(self, item_name)
                if current is not None:
                    held = [current] + held

            # 1. filter out Nones
            held = [el for el in held if el is not None]
            if len(held) == 0:
                continue

            # 2. work out if numpy or tensor mode (and check not mixed)
            all_np = [isinstance(el, np.ndarray) for el in held]
            if (not all(all_np)) and any(all_np):
                raise RuntimeError("Cannot mix numpy arrays and tensors in predictions output.")
            cat_op = np.concatenate if all_np else torch.cat

            # 3. work out maximum shape at dim 1 (if applicable) and pad all to that. if only one dim do not need to pad.
            dims = set([len(el.shape) for el in held])
            assert len(dims) == 1, "All held items must have same number of dimensions."
            dims = dims.pop()

            if dims == 1:
                padded = held
            else:
                max_dim = max([el.shape[1] for el in held])
                def pad(el):
                    # pad dim 1 at the outer edge up to max dim and leave all other dimensions untouched.
                    assert len(el.shape) >= 2
                    if isinstance(el, np.ndarray):
                        pd = [(0, 0), (0, max_dim - el.shape[1])] + [(0, 0)] * (len(el.shape) - 2)
                        return np.pad(el, tuple(pd), mode='constant', constant_values=pad_value)
                    else:
                        pd = [0, max_dim - el.shape[1]] + [0] * ((len(el.shape) - 2) * 2)
                        return F.pad(el, tuple(pd), value=pad_value, mode='constant')
                padded = [pad(el) for el in held]
            del held  # just because otherwise might keep up a lot of memory...

            concatenated = cat_op(padded)
            del padded

            setattr(self, item_name, concatenated)
            delattr(self, held_name)

    def _get_held_name_from_prop_name(self, prop_name):
        return self.HELD_OTHERS_PREFIX + prop_name

    @property
    def _property_names_to_join(self):
        return [el[0] for el in self._property_name_and_pad_value]

    @property
    def _property_name_and_pad_value(self):
        return (("pseudo_per_item_losses", -100),
                ("generated_tokens", -100),
                ("labels", -100),
                ("manually_computed_losses_per_item_per_token", 0.),
                ("encoder_attn_zerod_outputs", 0.),
                ("input_ids", self.pad_token_input_id)  # <- we're using 1 as the integer id for pad tokens, change
                # if change vocab.
                )

    @property
    def is_np(self):
        if self._is_np is not None:
            return self._is_np
        else:
            return isinstance(self.pseudo_per_item_losses, np.ndarray)

    def _cpufy_and_numpify_if_appropriate(self, possible_tensor):
        """
        If `possible_tensor` is a Pytorch Tensor move to cpu. Furthermore, if class `is_np` is True then convert
        `possible_tensor` to numpy array. (if already a numpy array do nothing).
        """
        if isinstance(possible_tensor, torch.Tensor):
            possible_tensor = possible_tensor.cpu()
            if self.is_np:
                possible_tensor = possible_tensor.numpy()
        return possible_tensor

    def _pad_and_concat(self, first, second, pad_value=-100):
        """
        Pads (on dim=1) and concats on dim=0. Agnostic to whether first/second are np arrays or pt tensors (although it
        has to be consistent).
        """
        #depreciated but might make a nice test.
        if first is None and second is None:
            return None

        is_np_op = isinstance(first, np.ndarray)
        if is_np_op: assert isinstance(second, np.ndarray), "Types do not match."

        if len(first.shape) == 1 or first.shape[1] == second.shape[1]:
            cat_op = np.concatenate if is_np_op else torch.cat
            return cat_op((first, second))
        else:
            new_shape = (first.shape[0] + second.shape[0], max(first.shape[1], second.shape[1])) + first.shape[2:]

            if is_np_op:
                out = np.full_like(first, pad_value, shape=new_shape)
            else:
                out = first.new_full(new_shape, pad_value)

            out[:first.shape[0], :first.shape[1]] = first
            out[first.shape[0]:, :second.shape[1]] = second
            return out


def prediction_step(model: nn.Module, inputs: dict, args: PredictionStepArgs) -> PredictionOutput:
    """
    A lot of this function is based off the `prediction_step` method of the Transformer library's `Seq2SeqTrainer`.

    Returns `PredictionOutput` which contains:
        - `loss` mean loss
        - `generated_tokens`:
            * if `args.teacher_forcing` done by taking the argmax of the model's logits, and when
                feeding in the `decoder_input_ids` can think of as teacher forcing.
            * if not `args.teacher_forcing` then will generate the tokens using the models `generate` method (e.g.,
                beam_search), args.num_beams should be set and the number of predictions returned for each point
                are set in the model config so note that the `generated_tokens` tensor size may be different to when
                using teacher forcing.
        - `labels`: labels tensor from the inputs dict (if given).

    """
    # # 1. If not teacher forcing generate the output by using the model's `generate` method.
    if not args.teacher_forcing:
        gen_kwargs = {
            "max_length": args.max_length if args.max_length is not None else model.config.max_length,
            "num_beams": args.num_beams if args.num_beams is not None else model.config.num_beams,
            "num_return_sequences": args.num_return_sequences if args.num_return_sequences is not None
                                                                        else model.config.num_return_sequences,
            "synced_gpus": False,
        }
        # some encoder-decoder models can have varying encoder's and thus
        # varying model input names
        if hasattr(model, "encoder") and model.encoder.main_input_name != model.main_input_name:
            generation_inputs = inputs[model.encoder.main_input_name]
            warnings.warn(f"Using inputs from encoder ({model.encoder.main_input_name}) "
                          f"rather than model ({model.main_input_name})")
        else:
            generation_inputs = inputs[model.main_input_name]
        generated_tokens = model.generate(
            generation_inputs,
            attention_mask=inputs.get("attention_mask", None),
            **gen_kwargs,
        )
    else:
        generated_tokens = None

    # # 2. Run the model in the forward manner
    has_labels = "labels" in inputs
    sequence_losses = None  # <- by default won't compute this and will fill in only if have labels and flag to compute
    # is set.

    # ## 2a. If has labels then run the method in forward manner and get loss, generated_tokens (if applicable), and
    # labels
    if has_labels:
        # run model forward
        with torch.no_grad():
            outputs = model(**inputs)

        # extract losses
        losses = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).detach()

        # get the generated tokens if currently None
        if generated_tokens is None:
            logits = outputs["logits"].detach()
            generated_tokens = logits.argmax(dim=-1)

        # get labels
        labels = inputs["labels"]

        # manually compute per item per token losses. Hugging Face returns the mean, but some analysis require the
        # actual per item losses. Useful for knowing
        if args.manually_compute_per_item_losses:
            loss_fct = nn.CrossEntropyLoss(reduction="none")
            logits_shape = outputs["logits"].shape
            sequence_losses = loss_fct(outputs["logits"].view(-1, logits_shape[-1]),
                                                                      labels.view(-1)).view(logits_shape[:2]).detach()
            loss_diff = torch.abs((sequence_losses.sum() / (labels != -100).sum()) - losses)
            assert torch.all(loss_diff < 0.001), f"Losses do not match, diff {loss_diff}"
            # ^ just check that the average is what we expect --

    # ## 2b. If does not have labels then only run forward to get the generated tokens (if currently None) -- losses
    #           and labels will be None
    else:
        losses = None
        labels = None
        if generated_tokens is None:
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs["logits"]
                generated_tokens = logits.argmax(dim=-1).detach()

    # # 3. collect other tensors from inputs/outputs if required.
    if args.collect_encoder_outputs_type:
        if not isinstance(outputs, dict):
            raise RuntimeError("Cannot extract encoder outputs as model outputs is not a dict (check model config!)")
        with torch.no_grad():
            encoder_last_hidden = outputs["encoder_last_hidden_state"].clone()
            encoder_last_hidden[~inputs["attention_mask"].unsqueeze(-1).expand(*encoder_last_hidden.shape).bool()] = 0
            encoder_last_hidden = encoder_last_hidden.detach()
            if args.encoder_embeddings_op is not None:
                encoder_last_hidden = args.encoder_embeddings_op(encoder_last_hidden)
    else:
        encoder_last_hidden = None

    if args.collect_input_ids:
        input_ids = inputs["input_ids"].detach()
    else:
        input_ids = None

    # # 4. Pad variable length outputs to max lengths
    generated_tokens, labels = [pad_last(el, args.max_length, model.config.pad_token_id) if el is not None else None
                                for el in [generated_tokens, labels]]

    # # 5. Form output
    out = PredictionOutput(losses, generated_tokens, labels,
                           sequence_losses, encoder_last_hidden, input_ids, model.config.pad_token_id)
    return out


def eval_loop(dataloader, model, prepare_inputs,
              prediction_step_args: PredictionStepArgs, use_tqdm=True) -> Tuple[PredictionOutput, int]:
    # # 1. Setup model for evaluation mode
    model.eval()

    # # 2. Iterate through dataloader
    observed_num_examples = 0
    prediction_outputs = PredictionOutput(_is_np=True)

    iterator = tqdm(dataloader, desc="evaluating...") if use_tqdm else dataloader
    for step, inputs in enumerate(iterator):

        observed_batch_size = trainer_pt_utils.find_batch_size(inputs)
        if observed_batch_size is not None:
            observed_num_examples += observed_batch_size

        inputs, _ = prepare_inputs(inputs)
        op = prediction_step(model, inputs, prediction_step_args)
        op.repeat_losses(observed_batch_size)
        # ^ we will pretend each member of the batch has the same loss to make averaging easier later!
        prediction_outputs.hold_others_but_do_not_cat_or_pad(op)
    prediction_outputs.pad_and_concat_held_others()

    return prediction_outputs, observed_num_examples


def create_metrics_compute_func(tokenizer, clean_pred_func, log=None):
    def compute_metrics(prediction_outputs: PredictionOutput, observed_num_examples):
        metrics_out = {}

        # # 1. Loss
        if prediction_outputs.pseudo_per_item_losses is not None:
            loss = prediction_outputs.pseudo_per_item_losses.mean()
            assert prediction_outputs.pseudo_per_item_losses.shape[0] == observed_num_examples
            metrics_out["loss"] = loss

        # # 2. Accuracy/ Gen Lengths
        if prediction_outputs.generated_tokens is not None:
            assert prediction_outputs.generated_tokens.shape[0] % observed_num_examples == 0
            num_return_sequences = int(prediction_outputs.generated_tokens.shape[0] / observed_num_examples)
            metrics_out["num_inferred_return_sequences"] = num_return_sequences

            labels = prediction_outputs.labels
            # Replace -100 in the labels as we can't decode them:
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            if log is not None:
                log.debug("about to batch decode...")
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
            if log is not None:
                log.debug("finished batch decode (labels)")

            generated_tokens = prediction_outputs.generated_tokens
            generated_tokens = np.where(generated_tokens != -100, generated_tokens, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            if log is not None:
                log.debug("finished batch decode (preds)")

            accuracies = np.zeros((observed_num_examples, num_return_sequences))
            gen_smile_lengths = -np.ones((observed_num_examples, num_return_sequences))
            gen_lengths = np.sum(generated_tokens != tokenizer.pad_token_id, axis=-1).reshape(
                (observed_num_examples, num_return_sequences))

            for i, decoded_label in enumerate(tqdm(decoded_labels, desc="checking accuracies.")):
                canonical_label = clean_pred_func(decoded_label)
                for j in range(num_return_sequences):
                    pred = decoded_preds[i*num_return_sequences + j]
                    canonical_pred = clean_pred_func(pred)

                    gen_smile_lengths[i, j] = len(pred)
                    accuracies[i, j] = canonical_pred == canonical_label

            top_k_accuracies = np.mean(np.cumsum(accuracies, axis=1) > 0.5, axis=0)
            gen_smile_lengths = np.mean(gen_smile_lengths, axis=0)
            gen_lengths = np.mean(gen_lengths, axis=0)
            for i in range(num_return_sequences):
                # although we index from 0 in Python, it is more natural to index from 1 when describing top-k
                # accuracies in natural language
                metrics_out[f"top-{i + 1}_accuracy"] = top_k_accuracies[i]
                metrics_out[f"top-{i + 1}_genlength"] = gen_lengths[i]
                metrics_out[f"top-{i + 1}_gensmileslength"] = gen_smile_lengths[i]
        else:
            decoded_preds = None

        return metrics_out, decoded_preds
    return compute_metrics


def pad_last(tensor, max_length, pad_value):
    if (tensor is not None) and (tensor.shape[-1] < max_length):
        return F.pad(tensor, (0, max_length - tensor.shape[-1]), mode='constant', value=pad_value)
    else:
        return tensor

