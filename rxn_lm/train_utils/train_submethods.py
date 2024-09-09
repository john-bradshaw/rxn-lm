"""
As mentioned in the comments below, some of this code is adapted from PyTorch Ignite, this has the following licence:

BSD 3-Clause License

Copyright (c) 2018-present, PyTorch-Ignite team
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Some also from the Transformer's library, with licence:
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
import enum
import json
from typing import (
    Any,
    Mapping,
    Union,
    Callable,
    Optional,
    Sequence,
    Tuple,
    cast,
)

import torch
from ignite import engine, base
from ray import tune
from torch import nn
from torch import optim
from torch.utils import data
from transformers import trainer_pt_utils

from rxn_lm.settings import REACTANTS_KEY, PRODUCTS_KEY


def tune_search_space_from_json(json_in):
    """
    Converts a json in to a tune search space. The json should be a dictionary of dictionaries.
    The keys in the outer dictionary specify the name of the parameter to tune.
    The inner dictionary should contain the following keys:
        type: with value of str, one of ["choice", "uniform", "loguniform", "randint"]
        values: with value of list, containing the values to choose from if type is "choice", and if one of other types
            the values that should get unpacked.
    """
    out_dict = {}
    for key, value in json_in.items():
        type_ = value["type"]
        if type_ == "choice":
            out_dict[key] = tune.choice(value["values"])
        elif type_ == "uniform":
            out_dict[key] = tune.uniform(*value["values"])
        elif type_ == "loguniform":
            out_dict[key] = tune.loguniform(*value["values"])
        elif type_ == "randint":
            out_dict[key] = tune.randint(*value["values"])
        else:
            raise ValueError(f"Unknown type {value['type']}")
    return out_dict


class EarlyStopper(base.Serializable):
    """
    Early stopping handler based on the Ignite one so that it can load checkpoints,
    but does not operate on an Engine.

    Based on `ignite/handlers/early_stopping.py`

    To use call when you evaluate each score. If not improving (i.e., going up) then will terminate the trainer.
    """
    _state_dict_all_req_keys = (
        "counter",
        "best_score",
    )

    def __init__(
            self,
            logger,
            patience: int,
            trainer_to_terminate: engine.Engine,
            min_delta: float = 0.0,
            cumulative_delta: bool = False,
    ):
        """
        :param logger: run logger
        :param patience: Number of events to wait before stopping when no improvement is found.
        :param trainer_to_terminate: The Ignite trainer to terminate.
        :param min_delta: A minimum increase in the score to qualify as an improvement,
            i.e. an increase of less than or equal to `min_delta`, will count as no improvement.
        :param cumulative_delta: If True, `min_delta` defines an increase since the last `patience` reset, otherwise,
            it defines an increase after the last event. Default value is False.
        """
        if patience < 1:
            raise ValueError("Argument patience should be positive integer.")

        if min_delta < 0.0:
            raise ValueError("Argument min_delta should not be a negative number.")

        if not isinstance(trainer_to_terminate, engine.Engine):
            raise TypeError("Argument trainer should be an instance of Engine.")

        self.patience = patience
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta
        self.trainer_to_terminate = trainer_to_terminate
        self.logger = logger

        self.iteration_last_called = -1
        self.counter = 0
        self.best_score: Optional[float] = None

    def __call__(self, score, iteration):
        """
        :param score: higher score is better.
        :param iteration: current iteration. (will ignore score if not from a later iteration).
        """
        if iteration > self.iteration_last_called:
            self.iteration_last_called = iteration
        else:
            return

        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.min_delta:
            if not self.cumulative_delta and score > self.best_score:
                self.best_score = score
            self.counter += 1
            self.logger.debug(f"Early stopping: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.logger.info(f"Early stopping: Stop training (score: {score}, best_score: {self.best_score})")
                self.trainer_to_terminate.terminate()
        else:
            self.best_score = score
            self.counter = 0

    def state_dict(self) -> "collections.OrderedDict[str, float]":
        """Method returns state dict with ``counter`` and ``best_score``.
        Can be used to save internal state of the class.
        """
        return collections.OrderedDict([("counter", self.counter),
                                        ("best_score", cast(float, self.best_score)),
                                        ("iteration_last_called", cast(int, self.iteration_last_called))])

    def load_state_dict(self, state_dict: Mapping) -> None:
        """Method replace internal state of the class with provided state dict data.

        Args:
            state_dict: a dict with "counter" and "best_score" keys/values.
        """
        super().load_state_dict(state_dict)
        self.counter = state_dict["counter"]
        self.best_score = state_dict["best_score"]
        self.iteration_last_called = state_dict["iteration_last_called"]


class TaskDirection(enum.Enum):
    FORWARDS = "forward"  # i.e., reaction prediction
    BACKWARDS = "backward"  # i.e., one-step retro-synthesis, have not actually used the model for this!


class ReactionData(data.Dataset):
    def __init__(self, data_lines, transform=None):
        self.data_lines = data_lines
        self.transform = transform

    def __getitem__(self, idx):
        out = {"idx": idx, **self.data_lines[idx]}
        if self.transform is not None:
            out = self.transform(out)
        return out

    def __len__(self):
        return len(self.data_lines)

    def select(self, indcs_to_select):
        new_data = [self.data_lines[i] for i in indcs_to_select]
        return self.__class__(new_data, self.transform)

    @classmethod
    def from_smiles_file(cls, smiles_file_path, add_reagents_to_reactants=True):
        """
        :param smiles_file_path: path to a file containing the reactions as SMILES strings, e.g. reactants>>products
                                    or reactants>reagents>products
        :param add_reagents_to_reactants: add the reagents (if defined) to the reactants; otherwise discard. Note that
        this does not shuffle the LH side so add a shuffle transform if you want that.
        :return: ReactionData class
        """
        with open(smiles_file_path, 'r') as fo:
            smiles_lines = [e.strip() for e in fo.readlines()]
            smiles_lines = [e for e in smiles_lines if len(e)]

        data_lines = []
        for smiles_string in smiles_lines:
            reactants, reagents, products = smiles_string.split('>')
            if add_reagents_to_reactants and len(reagents):
                reactants += "." + reagents
            data_lines.append(dict(
                translation=dict(REACTANTS_KEY=reactants, PRODUCTS_KEY=products)
            ))
        return cls(data_lines)

    @classmethod
    def from_jsonl_file(cls, json_lines_file_path):
        with open(json_lines_file_path, 'r') as fo:
            smiles_lines = [e.strip() for e in fo.readlines()]
            smiles_lines = [json.loads(e) for e in smiles_lines if len(e)]
        return cls(smiles_lines)


def create_preprocess_func(tokenizer, max_length, padding, ignore_pad_token_for_loss,
                           direction=TaskDirection.FORWARDS, prefix=""):
    if direction is TaskDirection.FORWARDS:
        src_key = REACTANTS_KEY
        tgt_key = PRODUCTS_KEY
    else:
        src_key = PRODUCTS_KEY
        tgt_key = REACTANTS_KEY

    def preprocess_function(example):
        inputs = example["translation"][src_key]
        targets = example["translation"][tgt_key]
        inputs = prefix + inputs
        model_inputs = tokenizer(inputs, max_length=max_length, truncation=True)

        # Set up the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_length, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess_function


def get_optimizer(model, learning_rate, weight_decay):
    decay_parameters = set([name for name in
                        trainer_pt_utils.get_parameter_names(model, [nn.LayerNorm])
                        if "bias" not in name])
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    optimizer = optim.AdamW(optimizer_grouped_parameters, learning_rate)
    # there is actually also a version of AdamW in HuggingFace: https://github.com/huggingface/transformers/issues/3407
    return optimizer


def prepare_input(data, device):
    # based on the Transformers training version (minus deepseed stuff)
    if isinstance(data, Mapping):
        return type(data)({k: prepare_input(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(prepare_input(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = dict(device=device)
        return data.to(**kwargs)
    return data


def grad_clipping(model, max_grad_norm, optimizer):
    # based on the one in Transformer's trainer (minus deepseed and grad scaling/amp stuff).
    # comments below are from the respective place in the transformers library

    if max_grad_norm is not None and max_grad_norm > 0:
        if hasattr(optimizer, "clip_grad_norm"):
            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
            optimizer.clip_grad_norm(max_grad_norm)
        elif hasattr(model, "clip_grad_norm_"):
            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
            model.clip_grad_norm_(max_grad_norm)
        else:
            # Revert to normal clipping otherwise, handling Apex or full precision
            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )


def create_supervised_trainer(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Union[Callable, torch.nn.Module],
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False,
        prepare_batch: Callable = engine._prepare_batch,
        output_transform: Callable[[Any, Any, Any, torch.Tensor], Any] = lambda x, y, y_pred, loss: loss.item(),
        deterministic: bool = False,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 0  # <- 0 means none
    ) -> engine.Engine:
    """
    from Ignite minus the amp.apex/tpu stuff
    but with addition of gradient clipping.
    """
    _update = supervised_training_step(
        model,
        optimizer,
        loss_fn,
        device,
        non_blocking,
        prepare_batch,
        output_transform,
        gradient_accumulation_steps,
        max_grad_norm
    )

    trainer = engine.Engine(_update) if not deterministic else engine.DeterministicEngine(_update)
    return trainer


def supervised_training_step(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Union[Callable, torch.nn.Module],
        device: Optional[Union[str, torch.device]] = None,
        non_blocking: bool = False,
        prepare_batch: Callable = engine._prepare_batch,
        output_transform: Callable[[Any, Any, Any, torch.Tensor], Any] = lambda x, y, y_pred, loss: loss.item(),
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 0  # <- 0 means none
    ) -> Callable:
    """
    from Ignite with addition of gradient clipping.
    """

    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "Gradient_accumulation_steps must be strictly positive. "
            "No gradient accumulation if the value set to one (default)."
        )

    def update(engine: engine.Engine, batch: Sequence[torch.Tensor]) -> Union[Any, Tuple[torch.Tensor]]:
        model.train()
        if (engine.state.iteration - 1) % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
        try:
            y_pred = model(**x)
        except Exception as ex:
            print("batch that caused the error")
            print(batch)
            try:
                for k,v in batch.items():
                    try:
                        print(f"{k}: {v.shape}")
                    except:
                        pass
            except:
                pass
            raise ex
        loss = loss_fn(y_pred, y)
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        loss.backward()

        if engine.state.iteration % gradient_accumulation_steps == 0:
            grad_clipping(model, max_grad_norm, optimizer)
            optimizer.step()
        return output_transform(x, y, y_pred, loss)

    return update