# Copyright 2024 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from abc import ABC, abstractmethod

import mlrun


class Application(ABC):
    def __init__(self, name: str, project_name: str = None, *args, **kwargs):
        self._name = name
        self._project = (
            mlrun.get_or_create_project(project_name)
            if project_name
            else mlrun.get_current_project()
        )
        self._application_runtime = None

    @abstractmethod
    def deploy(self, *args, **kwargs):
        pass

    @abstractmethod
    def invoke(self, *args, **kwargs) -> dict:
        pass

    def is_deployed(self):
        return self._application_runtime is not None
