import labelbox
LB_API_KEY = ''
PROJECT_ID = 'clqi3q3uj0awi07ycbz1d8ul1'
client = labelbox.Client(api_key = LB_API_KEY)
project = client.get_project(PROJECT_ID)
labels = project.export_v2(params={
	"data_row_details": True,
	"metadata_fields": True,
	"attachments": True,
	"project_details": True,
	"performance_details": True,
	"label_details": True,
	"interpolated_frames": True
  })