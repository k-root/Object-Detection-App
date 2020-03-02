import { async, ComponentFixture, TestBed } from '@angular/core/testing';

import { TestEvaluateModelComponent } from './test-evaluate-model.component';

describe('TestEvaluateModelComponent', () => {
  let component: TestEvaluateModelComponent;
  let fixture: ComponentFixture<TestEvaluateModelComponent>;

  beforeEach(async(() => {
    TestBed.configureTestingModule({
      declarations: [ TestEvaluateModelComponent ]
    })
    .compileComponents();
  }));

  beforeEach(() => {
    fixture = TestBed.createComponent(TestEvaluateModelComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
