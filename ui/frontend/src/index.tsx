/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @ts-ignore
import React from 'react'
// @ts-ignore
import ReactDOM from "react-dom"


import {
  ComponentProps,
  withStreamlitConnection,
    // @ts-ignore
} from 'streamlit-component-lib'


import ContributionGraph from "./ContributionGraph"
import Selector from "./Selector"

const LlmViewerComponent = (props: ComponentProps) => {
  switch (props.args['component']) {
    case 'graph':
      // @ts-ignore
      return <div class="scrollmenu"><ContributionGraph /></div>
    case 'selector':
      return <Selector />
    default:
      return <></>
  }
};

const StreamlitLlmViewerComponent = withStreamlitConnection(LlmViewerComponent)

ReactDOM.render(
  <React.StrictMode>
    <StreamlitLlmViewerComponent />
  </React.StrictMode>,
  document.getElementById("root")
)
