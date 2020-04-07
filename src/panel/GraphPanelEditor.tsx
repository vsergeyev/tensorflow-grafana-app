// Libraries
import _ from 'lodash';
import React, { PureComponent } from 'react';

// Services
import { getDataSourceSrv } from '@grafana/runtime';

// Types
import { PanelEditorProps, FieldConfig, DataSourceSelectItem } from '@grafana/data';
import {
  Switch,
  LegendOptions,
  GraphTooltipOptions,
  PanelOptionsGrid,
  PanelOptionsGroup,
  FieldPropertiesEditor,
  Select,
  Input,
} from '@grafana/ui';


import { Options, GraphOptions, GraphDatasourceOptions } from './types';
import { GraphLegendEditor } from './GraphLegendEditor';

export class GraphPanelEditor extends PureComponent<PanelEditorProps<Options>> {
  constructor(props) {
    super(props)
    // window.console.log("GraphPanelEditor", props);
  }

  onChangeEpochs = (event: any) => {
    this.props.options.datasourceOptions.epochs = event.target.value;
    this.setState({value: event.target.value});
  };

  onBlurEpochs = () => {
    // window.console.log("onBlurInputBucket", this.state);
  };

  onChangeBatchSize = (event: any) => {
    this.props.options.datasourceOptions.batchsize = event.target.value;
    this.setState({value: event.target.value});
  };

  onBlurBatchSize = () => {
    // window.console.log("onBlurOutputBucket", this.state);
  };

  onGraphOptionsChange = (options: Partial<GraphOptions>) => {
    this.props.onOptionsChange({
      ...this.props.options,
      graph: {
        ...this.props.options.graph,
        ...options,
      },
    });
  };

  onLegendOptionsChange = (options: LegendOptions) => {
    this.props.onOptionsChange({ ...this.props.options, legend: options });
  };

  onTooltipOptionsChange = (options: GraphTooltipOptions) => {
    this.props.onOptionsChange({ ...this.props.options, tooltipOptions: options });
  };

  onToggleLines = () => {
    this.onGraphOptionsChange({ showLines: !this.props.options.graph.showLines });
  };

  onToggleBars = () => {
    this.onGraphOptionsChange({ showBars: !this.props.options.graph.showBars });
  };

  onTogglePoints = () => {
    this.onGraphOptionsChange({ showPoints: !this.props.options.graph.showPoints });
  };

  onToggleisStacked = () => {
    this.onGraphOptionsChange({ isStacked: !this.props.options.graph.isStacked });
  }

  onChangeFill = (value: any) => {
    this.onGraphOptionsChange({ fill: value.value });
    this.setState({ value: value.value });
  };

  onChangeFillGradient = (value: any) => {
    this.onGraphOptionsChange({ fillGradient: value.value });
    this.setState({ value: value.value });
  };

  onChangeLineWidth = (value: any) => {
    this.onGraphOptionsChange({ lineWidth: value.value });
    this.setState({ value: value.value });
  };

  onDefaultsChange = (field: FieldConfig) => {
    this.props.onOptionsChange({
      ...this.props.options,
      fieldOptions: {
        ...this.props.options.fieldOptions,
        defaults: field,
      },
    });
  };

  render() {
    const {
      graph: { showBars, showPoints, showLines, isStacked, lineWidth, fill, fillGradient },
      tooltipOptions: { mode },
      datasourceOptions: { epochs, batchsize },
    } = this.props.options;

    return (
      <>
        <PanelOptionsGroup title="Machine Learning">
          <div className="gf-form max-width-40">
            <span className="gf-form-label width-10">Epochs</span>
            <Input
              value={this.props.options.datasourceOptions.epochs}
              className="gf-form-input" type="text"
              placeholder="Epochs for training"
              min-length="0"
              onBlur={this.onBlurEpochs}
              onChange={this.onChangeEpochs}
            />
          </div>

          <div className="gf-form max-width-40">
            <span className="gf-form-label width-10">Batch size</span>
            <Input
              value={this.props.options.datasourceOptions.batchsize}
              className="gf-form-input" type="text"
              placeholder="Size of batch"
              min-length="0"
              onBlur={this.onBlurBatchSize}
              onChange={this.onChangeBatchSize}
            />
          </div>
        </PanelOptionsGroup>

        <PanelOptionsGroup title="Draw">
          <div className="section gf-form-group">
            <h5 className="section-heading">Draw Modes</h5>
            <Switch label="Lines" labelClass="width-5" checked={showLines} onChange={this.onToggleLines} />
            <Switch label="Bars" labelClass="width-5" checked={showBars} onChange={this.onToggleBars} />
            <Switch label="Points" labelClass="width-5" checked={showPoints} onChange={this.onTogglePoints} />
          </div>
          <div className="section gf-form-group">
            <h5 className="section-heading">Mode Options</h5>
            <div className="gf-form max-width-20">
              <span className="gf-form-label width-8">Fill</span>
              <Select
                className="width-5"
                value={{ value: fill, label: fill }}
                onChange={value => {
                  this.onChangeFill({ value: value.value as any });
                }}
                options={[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(t=>({value: t, label: t}))}
                />
            </div>
            <div className="gf-form max-width-20">
              <span className="gf-form-label width-8">Fill Gradient</span>
              <Select
                className="width-5"
                value={{ value: fillGradient, label: fillGradient }}
                onChange={value => {
                  this.onChangeFillGradient({ value: value.value as any });
                }}
                options={[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(t=>({value: t, label: t}))}
                />
            </div>

            <div className="gf-form max-width-20">
              <span className="gf-form-label width-8">Line Width</span>
              <Select
                className="width-5"
                value={{ value: lineWidth, label: lineWidth }}
                onChange={value => {
                  this.onChangeLineWidth({ value: value.value as any });
                }}
                options={[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(t=>({value: t, label: t}))}
                />
            </div>
          </div>
          <div className="section gf-form-group">
            <h5 className="section-heading">Stacking & Null value</h5>
            <div className="gf-form max-width-20">
              <Switch label="Stack" labelClass="width-5" checked={isStacked} onChange={this.onToggleisStacked} />
            </div>
          </div>
        </PanelOptionsGroup>

        <PanelOptionsGrid>
          <PanelOptionsGroup title="Field">
            <FieldPropertiesEditor
              showMinMax={false}
              onChange={this.onDefaultsChange}
              value={this.props.options.fieldOptions.defaults}
            />
          </PanelOptionsGroup>
          <PanelOptionsGroup title="Tooltip">
            <Select
              value={{ value: mode, label: mode === 'single' ? 'Single' : 'All series' }}
              onChange={value => {
                this.onTooltipOptionsChange({ mode: value.value as any });
              }}
              options={[
                { label: 'All series', value: 'multi' },
                { label: 'Single', value: 'single' },
              ]}
            />
          </PanelOptionsGroup>
          <GraphLegendEditor options={this.props.options.legend} onChange={this.onLegendOptionsChange} />
        </PanelOptionsGrid>
      </>
    );
  }
}
